# results/

One folder per run, named after the config that produced it.

```
<run>/
  trajectory.csv   estimated poses: Timestamp,Pred_X,Pred_Y,Pred_Z,Q_X,Q_Y,Q_Z,Q_W
  run.log          console output of that run
  run.jsonl        structured log -- read with scripts/logreader.py, never raw
  eval/            everything derived from trajectory.csv
    ape.txt        evo metrics, no alignment (start-anchored, yaw only)
    gt.tum dr.tum aligned.tum evo.csv
    ape_startaligned_map.png
```

Nothing belongs at the top level of results/ except this file.

Scoring (compare_frl needs matplotlib for --plot, so use ~/venv/bin/python):

  ~/venv/bin/python scripts/compare_frl.py results --plot   # every run + graphs
  ~/venv/bin/python scripts/compare_frl.py results/<run>    # one run
  scripts/evaluate_mun3_frl.sh results/<run>/trajectory.csv # evo, yaw-only
  python3 scripts/logreader.py summary results/<run>/run.jsonl

compare_frl pairs VO to GT by TIMESTAMP and re-origins both at the first
matched sample. It and the evo path use the same start-anchored convention but
different resampling, so their absolute numbers are close but not identical --
compare within one scorer, never across.
