#!/usr/bin/env bash
# Parte 2 (Trilha C): orçamentos 1, 5 (3 seeds) e 20, mesmos hiperparâmetros.
set -e
cd "$(dirname "$0")/.."
for cfg in "5 43" "5 44" "1 42" "20 42"; do
  set -- $cfg
  ./venv/bin/python -m src.pseudo_labels --budget $1 --diagnose > runs/teacher_K$1.log 2>&1
  ./venv/bin/python -m src.distillation --budget $1 --seed $2 --epochs 8 --epoch-size 1536 \
    --checkpoint checkpoints/student_K$1_s$2.pth > runs/distill_K$1_s$2.log 2>&1
  echo "done K=$1 seed=$2"
done
