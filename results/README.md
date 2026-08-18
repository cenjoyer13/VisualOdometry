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

Scoring:
  scripts/compare_frl.py results/<run>        # project scorer, no alignment
  scripts/compare_frl.py results              # every run, one table
  scripts/evaluate_mun3_frl.sh results/<run>/trajectory.csv   # evo, yaw-only
  scripts/logreader.py summary results/<run>/run.jsonl
