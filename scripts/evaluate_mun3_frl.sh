#!/usr/bin/env bash
# Evaluate a RosbagEvaluator mun3 run against the MUN-FRL ground truth, using
# the evo_evaluate instrumentation and the START-ANCHORED convention the
# bell3/bell4 numbers were produced with:
#
#   1. strip the CSV header and normalise the timestamp column
#   2. prepare_evo.py - ENU-convert the FRL .pos, crop, downsample 1:1 to the
#                       5 Hz GT, rotate by the start yaw and snap the start
#   3. make_dr.py     - redo the yaw alignment as a yaw-only fit over the first
#                       30 m of GT motion, then snap the start anchor
#   4. evo_ape        - WITH NO ALIGNMENT, so accumulated drift is preserved
#
# Deliberately NOT evo's --align (Umeyama): a global 6-dof fit redistributes
# error across the whole trajectory and reports a much smaller number that says
# nothing about drift from the start. Yaw is the only free parameter, because
# it is the one quantity a gravity-aligned VIO genuinely cannot observe.
#
# Unlike VINS's own evaluate_mun3_frl.sh there is no quaternion reordering step:
# VINS writes qw,qx,qy,qz, while this project's evaluator already writes the
# qx,qy,qz,qw order prepare_evo.py expects (see PoseMath::rot2quat).
#
#   ./scripts/evaluate_mun3_frl.sh [trajectory.csv] [tag]
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVO_DIR=/home/lysenko/VO-testing/evo_evaluate
# evo_evaluate/evo_env is broken on this machine (ModuleNotFoundError: evo), so
# use the working venv from the VINS setup. Same evo, same convention.
PY="/home/lysenko/VO-testing/VINS/tools/evo_venv/bin/python"
EVO_APE="/home/lysenko/VO-testing/VINS/tools/evo_venv/bin/evo_ape"
GT_POS="$EVO_DIR/bell3/bell412_dataset3_frl.pos"

VIO=${1:-$HERE/results/mun3_optical/trajectory.csv}
TAG=${2:-}
# results/ is one folder per run; derive the tag from it and keep the derived
# artefacts in <run>/eval/ so the run folder stays readable.
if [ -z "$TAG" ]; then TAG="$(basename "$(dirname "$VIO")")"; fi
OUT="$(dirname "$VIO")/eval"
mkdir -p "$OUT"

for f in "$PY" "$EVO_APE" "$GT_POS" "$VIO"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

echo "[1/4] normalising CSV (drop header, keep qx,qy,qz,qw order)"
"$PY" - "$VIO" "$OUT/evo.csv" <<'EOF'
import sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src) as f, open(dst, 'w') as o:
    for line in f:
        p = [x for x in line.strip().split(',') if x != '']
        if len(p) < 8:
            continue
        try:
            t = float(p[0])
        except ValueError:
            continue                      # header row
        t = t / 1e9 if t > 1e17 else t
        x, y, z = p[1], p[2], p[3]
        qx, qy, qz, qw = p[4], p[5], p[6], p[7]
        o.write(f"{t:.6f},{x},{y},{z},{qx},{qy},{qz},{qw}\n")
        n += 1
print(f"      {n} poses")
EOF

echo "[2/4] prepare_evo.py (crop, 1:1 downsample to GT, start yaw + anchor)"
"$PY" "$EVO_DIR/prepare_evo.py" "$OUT/evo.csv" "$GT_POS" \
    --out_vins "$OUT/aligned.tum" --out_gt "$OUT/gt.tum" \
    | sed -n 's/^    -> /      /p'

echo "[3/4] make_dr.py (yaw-only fit over first 30 m, snap start)"
"$PY" "$EVO_DIR/make_dr.py" "$OUT/aligned.tum" "$OUT/gt.tum" \
    --out "$OUT/dr.tum" | sed 's/^/      /'

echo "[4/4] evo_ape, NO alignment"
"$EVO_APE" tum "$OUT/gt.tum" "$OUT/dr.tum" \
    --save_results "$OUT/ape.zip" \
    -p --plot_mode xy --save_plot "$OUT/ape_startaligned" \
    2>/dev/null | sed -n '/APE w.r.t/,/^$/p;/^ *\(max\|mean\|median\|min\|rmse\|sse\|std\)/p' \
    | tee "$OUT/ape.txt"

echo
echo "plot: $OUT/ape_startaligned_map.png"
