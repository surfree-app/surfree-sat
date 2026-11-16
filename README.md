# surfree-sat

The daily **beach check** for [Surfree](https://surfree.surfree.workers.dev), a free surf forecast.

Once a day a GitHub Action (`.github/workflows/beach.yml`) looks for new Sentinel-2 images of Surfree's twelve Portugal
spots (Ericeira, Peniche, Sines), reads only the 1.6 km near-infrared crop around each spot, skips cloudy or hazy ones,
and measures where waves break: the first-break line and the width of the white water. It compares that with the same
spot's own 2016–2026 history at the same sea state and tide (`ref/<slug>.json`) and sends the difference to the Surfree
worker, which flags a spot when its break sits persistently further out or closer in than usual ("sandbar moved").

Forecast heights are **not** changed by this check: on the 10-year archive the break distance did not predict the error of
the forecast height, so only the flag is used.

## Files

- `beachcheck.py` — the job. `python beachcheck.py --dry` runs it without sending anything.
- `ref/<slug>.json` — per spot: crop centre, pin, the history line `outer_m = c0 + c1·E + c2·tide` (E = the app's surf
  energy toward the spot), its scatter, and the settings the energy needs.
- `ref/<slug>.npz` — per spot: sea and land masks and the median near-infrared image of the crop (for the cloud screen).

The reference files were built from the archive by Surfree's `analysis/beach_check.py refs`; the measurement code here is
checked against that script image by image.

## Data and credit

- Sentinel-2 L2A from the AWS open data registry via Element84 Earth Search (no login). Contains modified Copernicus
  Sentinel data 2016–2026.
- Sea state (ERA5, Copernicus Climate Change Service) and sea level via [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0).

## Secret

`REFRESH_KEY` (repository Actions secret): the Surfree worker's admin key, used only for the one POST to `/api/beach`.
