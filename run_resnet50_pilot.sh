#!/bin/bash
# ResNet-50 counterflow pilot: 4 arms, 3 epochs, CIFAR-10, MPS.
#
# Launched detached (nohup + setsid) because the tool's background job list is
# cleared when the session worker restarts; a detached process keeps running.
#
# NOTE: warmup_epochs=5 in the configs exceeds the 3 epochs run here, so every
# arm trains under an identical truncated warmup ramp. That preserves the
# between-arm comparison; absolute accuracy will be below a full schedule.
set -u
cd "/Users/haosen/Developer/countercurrent"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cc_mpl}"
mkdir -p "$MPLCONFIGDIR" countercurrent_nn/results
LOG=countercurrent_nn/results/pilot_resnet50.log

echo "=== pilot start $(date) ===" >> "$LOG"
for cfg in resnet50_plain resnet50_lattice_control resnet50_countercurrent resnet50_cocurrent; do
  echo "" >> "$LOG"
  echo "######## $cfg ########" >> "$LOG"
  start=$(date +%s)
  ./.venv/bin/python -m countercurrent_nn.train \
    --config "countercurrent_nn/configs/cifar10_${cfg}.yaml" \
    --data-root ./dataset --no-download --device mps \
    --batch-size 128 --epochs 3 --num-workers 2 \
    --output-dir "countercurrent_nn/results/pilot_${cfg}" >> "$LOG" 2>&1
  rc=$?
  echo "exit=$rc for $cfg after $(( $(date +%s) - start ))s" >> "$LOG"
done
echo "" >> "$LOG"
echo "=== PILOT DONE $(date) ===" >> "$LOG"
