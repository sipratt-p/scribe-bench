#!/usr/bin/env bash
# Queue the expert-iteration run behind the plan-recall experiment; GPU 1 for the policy, vLLM :8004 for judging.
set -u
cd ~/projects/scribe-bench
source .venv/bin/activate
echo "waiting for plan_recall DONE $(date)"
until grep -q "^DONE" runs/plan_recall.log 2>/dev/null; do sleep 30; done
echo "plan_recall finished $(date); starting expert iteration"
python -m autoresearch.expert_iter run --rounds 4 --n 8 --k 2 --device 1
echo "EXIT $? $(date)"
