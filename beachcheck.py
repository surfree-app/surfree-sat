#!/usr/bin/env python3
"""Daily beach check for Surfree's Portugal spots, from new Sentinel-2 images (surfree-app/surfree card 6.16).

For each spot: find new clear Sentinel-2 L2A scenes (Element84 Earth Search STAC, AWS open data, no login), read only the
1.6 km near-infrared crop around the spot (HTTP range requests on the cloud-optimised GeoTIFF), screen clouds and haze,
and measure where waves break: the first-break line (75th percentile of the outermost foam along the spot's shore) and
the surf-zone width. The sea state at the image hour (ERA5 via Open-Meteo, ~30 km out) and the tide (Open-Meteo sea level)
give the break distance this spot usually shows in such a sea (ref/<slug>.json, fitted once on its 2016-2026 images).
The difference (residual, metres) is sent to the Surfree worker, which keeps 30 days of it per spot and raises a
"sandbar moved" flag when the break sits persistently further out or closer in than its history.

It does NOT change forecast heights: on the 10-year archive the break-distance residual did not predict the error of the
forecast height (Surfree notes/research/51-beach-check.md), so only the flag ships.

Usage: python beachcheck.py [--dry] [--days N]     env: SURFREE_URL (default the live worker), REFRESH_KEY (for the POST)
Credit: contains modified Copernicus Sentinel data; sea state and sea level by Open-Meteo.com (ERA5: Copernicus Climate
Change Service).
"""
import json, math, os, sys, time, urllib.error, urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {'User-Agent': 'surfree-study/0.1'}
SURFREE = os.environ.get('SURFREE_URL', 'https://surfree.surfree.workers.dev').rstrip('/')
STAC = 'https://earth-search.aws.element84.com/v1/search'
REF_VERSION = 1
os.environ.update(AWS_NO_SIGN_REQUEST='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                  GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_RETRY_DELAY='2', GDAL_HTTP_USERAGENT=UA['User-Agent'])

# ---------------------------------------------------------------- measurement (kept identical to Surfree's
# analysis/beach_check.py, which built ref/ from the archive; `beach_check.py refs` checks the two agree image by image)
FT, PX, HALF_PX, HALF_M = 0.3048, 10, 79.5, 800
FOAM, SEA, SEA_PCT, MAX_M, SWASH_M, MIN_PATCH, ZONE_M = 0.09, 0.035, 10, 700, 20, 4, 400


def angdiff(a, b): return abs(((a - b + 180) % 360 + 360) % 360 - 180)


def energy(sp, hs, T, d):
    """The app's surf energy (forecast.js surfMid): sqrt(w · H_ft² · T / 10), w = max(floor, cos(min(off, 90°)))."""
    w = max(sp.get('window_floor') if sp.get('window_floor') is not None else 0.25, math.cos(math.radians(min(angdiff(d, sp.get('swell_facing') or sp['facing']), 90))))
    return math.sqrt(w * (hs / FT) ** 2 * T / 10)


def geometry(sea, pin_rc):
    """distance (m) of each sea cell to shore, the shore cell it belongs to, and which cells count for this spot."""
    dist, (iy, ix) = ndimage.distance_transform_edt(sea, return_indices=True)
    dist = dist * PX
    yy, xx = np.mgrid[0:sea.shape[0], 0:sea.shape[1]]
    near = np.hypot(yy - pin_rc[0], xx - pin_rc[1]) * PX <= ZONE_M
    owner = iy * sea.shape[1] + ix                                   # flat index of the nearest shore cell
    band = sea & (dist > SWASH_M) & (dist <= MAX_M) & near
    return dist, owner, band


def clean(f):
    lab, n = ndimage.label(f, structure=np.ones((3, 3)))
    if not n: return f
    sizes = ndimage.sum(f, lab, range(1, n + 1)); return np.isin(lab, 1 + np.nonzero(sizes >= MIN_PATCH)[0])


def usable(nir, sea, land, med, dist):
    """Same screens as s2_breakzone.py: clouds brighten the land; haze/glint makes the calm sea beyond the surf zone bright."""
    if ((nir > med + 0.15) & land).sum() / max(1, land.sum()) > 0.01: return False
    calm = sea & (dist > MAX_M)
    if calm.sum() < 50: calm = sea & (dist > 500)
    return not (calm.any() and np.median(nir[calm]) > 0.05)


def measure(nir, dist, owner, band):
    """First-break line and surf-zone width in front of the spot's shore cells."""
    f = clean((nir > FOAM) & band)
    cells = np.unique(owner[band]); ncell = len(cells)
    if not f.any(): return dict(outer_m=None, width_m=0, share=0.0)
    o, d = owner[f], dist[f]
    keys, inv = np.unique(o, return_inverse=True)
    outer = np.zeros(len(keys)); np.maximum.at(outer, inv, d)
    cnt = np.bincount(inv)
    share = len(keys) / max(1, ncell)
    if len(keys) < 3: return dict(outer_m=None, width_m=int(np.median(cnt) * PX), share=round(share, 3))
    return dict(outer_m=int(np.percentile(outer, 75)), width_m=int(np.median(cnt) * PX), share=round(share, 3))
# ---------------------------------------------------------------- end of the shared part


def load_refs():
    refs = {}
    for f in sorted(os.listdir(os.path.join(HERE, 'ref'))):
        if not f.endswith('.json'): continue
        r = json.load(open(os.path.join(HERE, 'ref', f)))
        m = np.load(os.path.join(HERE, 'ref', f[:-5] + '.npz'))
        r['sea'], r['land'], r['med'] = m['sea'].astype(bool), m['land'].astype(bool), m['med'].astype(np.float32) / 10000
        r['dist'], r['owner'], r['band'] = geometry(r['sea'], r['pin_rc'])
        refs[r['slug']] = r
    return refs


def http_json(url, body=None, tries=4):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={**UA, **({'content-type': 'application/json'} if data else {})})
    for k in range(tries):
        try: return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429: raise
            if k == tries - 1: raise
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1: raise
        time.sleep(5 * (k + 1))


def scenes(ref, start, end):
    """New L2A items over the spot, oldest first (one search; the archive used the same catalogue)."""
    body = {'collections': ['sentinel-2-l2a'], 'intersects': {'type': 'Point', 'coordinates': [ref['crop'][1], ref['crop'][0]]},
            'datetime': f'{start}T00:00:00Z/{end}T23:59:59Z', 'query': {'eo:cloud_cover': {'lt': 30}}, 'limit': 100}
    return sorted(http_json(STAC, body).get('features', []), key=lambda i: i['properties']['datetime'])


def read_nir(item, ref):
    """The 160 × 160 NIR crop (reflectance), exactly as the archive crops were cut. Only a few range requests per image."""
    import rasterio
    from rasterio.warp import transform as warp
    from rasterio.windows import from_bounds
    lat, lon = ref['crop']
    with rasterio.open(item['assets']['nir']['href']) as src:
        x, y = warp('EPSG:4326', src.crs, [lon], [lat])
        win = from_bounds(x[0] - HALF_M, y[0] - HALF_M, x[0] + HALF_M, y[0] + HALF_M, src.transform)
        a = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
    off = 1000 if item['properties']['datetime'] >= '2022-01-25' else 0   # processing baseline 04.00 adds a 1000 offset
    if a.shape != (160, 160) or (a == 0).mean() > 0.2: return None        # off the tile edge
    return np.clip((a - off) / 10000, 0, 1)


_om = {}
def sea_at(ref, t):
    """ERA5 Hs, direction, mean period at the archive's ERA5 cell, and the tide at the archive's tide point, for hour t.
    ERA5 lands ~5 days late: returns None until it has, and the image is tried again on a later run."""
    day, hour = t[:10], t[:13] + ':00'
    la, lo = ref['era5']
    k1 = ('w', la, lo, day)
    if k1 not in _om:
        _om[k1] = http_json(f'https://marine-api.open-meteo.com/v1/marine?latitude={la}&longitude={lo}&start_date={day}&end_date={day}'
                            f'&hourly=wave_height,wave_direction,wave_period&models=era5_ocean&timezone=UTC&cell_selection=sea')['hourly']
    tl, to = ref['tide_point']
    k2 = ('t', tl, to, day)
    if k2 not in _om:
        _om[k2] = http_json(f'https://marine-api.open-meteo.com/v1/marine?latitude={tl}&longitude={to}&start_date={day}&end_date={day}'
                            f'&hourly=sea_level_height_msl&timezone=UTC&cell_selection=sea')['hourly']
    H, T = _om[k1], _om[k2]
    i = H['time'].index(hour) if hour in H['time'] else None
    j = T['time'].index(hour) if hour in T['time'] else None
    if i is None or j is None: return None
    v = (H['wave_height'][i], H['wave_direction'][i], H['wave_period'][i], T['sea_level_height_msl'][j])
    return None if None in v else v


def check_image(item, ref):
    """One image → a row for the worker: residual = measured break distance − the spot's usual one at this sea and tide."""
    t = item['properties']['datetime'][:16]
    nir = read_nir(item, ref)
    if nir is None: return dict(id=item['id'], t=t, skip='edge')
    if not usable(nir, ref['sea'], ref['land'], ref['med'], ref['dist']): return dict(id=item['id'], t=t, skip='cloud')
    s = sea_at(ref, t)
    if s is None: return None                                            # ERA5 not there yet: try again tomorrow
    hs, d, T, tide = s
    E = energy(ref['spot'], hs, T, d)
    m = measure(nir, ref['dist'], ref['owner'], ref['band'])
    c0, c1, c2 = ref['history']['c']
    res = None if m['outer_m'] is None or E < ref['history']['E_range'][0] else round(m['outer_m'] - (c0 + c1 * E + c2 * tide))
    return dict(id=item['id'], t=t, E=round(E, 2), tide=round(tide, 2), **m, res=res)


def main():
    args = sys.argv[1:]; dry = '--dry' in args
    days = int(args[args.index('--days') + 1]) if '--days' in args else 21
    refs = load_refs()
    today = datetime.now(timezone.utc).date()
    start, end = (today - timedelta(days=days)).isoformat(), (today - timedelta(days=1)).isoformat()
    try: known = http_json(f'{SURFREE}/api/beach', tries=2).get('spots', {})
    except Exception as e: print(f'no /api/beach state ({str(e)[:60]}): treating every image as new'); known = {}
    out, t0 = {}, time.time()
    for slug, ref in refs.items():
        seen = set(known.get(slug, {}).get('ids', [])); dates = set()
        rows = []
        for it in scenes(ref, start, end):
            day = it['properties']['datetime'][:10]
            if it['id'] in seen or day in dates: continue
            try: row = check_image(it, ref)
            except Exception as e: print(f'  {slug} {it["id"]}: {type(e).__name__} {str(e)[:80]}'); continue
            if row is None: continue
            if not row.get('skip'): dates.add(day)                    # one usable image per day, like the archive
            rows.append(row)
        if rows: out[slug] = dict(mad_m=ref['history']['mad_m'], rows=rows)
        print(f'{slug}: {len(rows)} new ({sum(1 for r in rows if r.get("res") is not None)} with a break line): '
              + ', '.join(f'{r["t"][:10]} {r.get("skip") or r.get("res")}' for r in rows), flush=True)
    print(f'read in {time.time() - t0:.0f} s')
    if not out: print('nothing new: no POST'); return
    body = dict(date=today.isoformat(), ref=REF_VERSION, spots=out)
    if dry: print(json.dumps(body)[:2000]); return
    key = os.environ.get('REFRESH_KEY')
    if not key: sys.exit('REFRESH_KEY missing')
    try: r = http_json(f'{SURFREE}/api/beach?key={key}', body, tries=2)
    except urllib.error.HTTPError as e:
        if e.code == 404: print('worker has no /api/beach yet (card 6.16 not deployed): nothing sent'); return
        raise
    print('worker:', json.dumps(r)[:500])


if __name__ == '__main__':
    main()
