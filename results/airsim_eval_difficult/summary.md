# AirSim playback sweep — summary

Logs: `results\airsim_eval_difficult\logs`  |  Plots: `results\airsim_eval_difficult\plots`

`RMSE raw` is the per-frame Euclidean error after start-locked alignment. `RMSE @ 10 Hz` resamples GT and VO to a common 10 Hz time grid (linear interp) before computing the error, so configs with very different frame rates contribute equally many samples to the average.

Sorted by `RMSE @ 10 Hz` (lower is better).

| # | Config | Frames | RMSE raw [m] | RMSE @ 10 Hz [m] | Mean FPS | Notes |
|---|--------|-------:|-------------:|---------------:|---------:|-------|
| 1 | `airsim_superpoint_flannmatch` | 1474 | 12.946 | 12.814 | 6.2 |  |
| 2 | `airsim_superpoint_flannmatch_lba` | 1481 | 16.142 | 17.277 | 6.2 |  |
| 3 | `airsim_sift_lightgluematch` | 835 | 11.767 | 18.130 | 3.8 |  |
| 4 | `airsim_superpoint_brutematch_lba` | 1894 | 19.185 | 19.048 | 8.0 |  |
| 5 | `airsim_superpoint_brutematch` | 1926 | 18.720 | 20.113 | 8.2 |  |
| 6 | `airsim_superpoint_lightglue_lba` | 407 | 17.501 | 28.286 | 1.6 |  |
| 7 | `airsim_sift_flannmatch` | 3996 | 34.874 | 33.234 | 18.5 |  |
| 8 | `airsim_superpoint_lightglue` | 409 | 14.211 | 34.339 | 1.7 |  |
| 9 | `airsim_sift_lightgluematch_lba` | 842 | 12.264 | 35.108 | 3.8 |  |
| 10 | `airsim_sift_flannmatch_lba` | 3887 | 74.354 | 35.144 | 18.2 |  |
| 11 | `airsim_sift_brutematch_lba` | 5427 | 41.255 | 39.931 | 26.6 |  |
| 12 | `airsim_sift_brutematch` | 5164 | 73.032 | 44.357 | 24.5 |  |
| 13 | `airsim_orb_brutematch_lba` | 4591 | 44.515 | 46.091 | 21.8 |  |
| 14 | `airsim_orb_brutematch` | 4576 | 91.858 | 54.351 | 21.8 |  |
