#!/usr/bin/env bash
# Build an anonymized release tree: code, prompts, lexicon, splits, cached outputs (small), wiki, paper; no machine names or tailnet addresses.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/scribe-bench-release}
rm -rf "$OUT"; mkdir -p "$OUT"
rsync -a --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'runs/*.wav' --exclude 'runs/bundle' --exclude 'runs/expert_iter/adapter_*' --exclude 'runs/expert_iter/samples_*' --exclude 'runs/live_synth/lookups' --exclude 'demo/uploads' --exclude 'demo/data' --exclude 'runs/*.log' --exclude 'runs/*.png' --exclude 'wiki/abridge.md' --exclude 'wiki/interview-narrative.md' --exclude 'wiki/nvidia-engagement.md' --exclude 'release' ./ "$OUT/"
# scrub identifiers
grep -rl -E '100\.83\.231\.108|100\.90\.251\.52|beast|cosmo|sethcosmo|sipratt|tail2f894f|Seth' "$OUT" --include='*.py' --include='*.md' --include='*.html' --include='*.sh' --include='*.json' --include='*.toml' 2>/dev/null | while read -r f; do
  sed -i '' -e 's/100\.83\.231\.108/GPU_HOST/g' -e 's/100\.90\.251\.52/DEMO_HOST/g' -e 's/seth-cosmo-studio\.tail2f894f\.ts\.net/DEMO_HOST/g' -e 's/sethcosmo/user/g' -e 's/sipratt@gmail\.com/author@example.org/g' -e 's/Seth Pratt/Anonymous/g' -e "s/Seth's/the reviewer's/g" -e 's/Seth:/Reviewer:/g' -e 's/Seth\b/the reviewer/g' -e 's/beast-ubn/gpubox/g' -e 's/\bbeast\b/gpubox/g' -e 's/\bcosmo\b/workstation/g' "$f"
done
echo "remaining identifiers:"; grep -rn -E '100\.83\.231\.108|100\.90\.251\.52|sethcosmo|sipratt|tail2f894f' "$OUT" | head -n 5 || true
du -sh "$OUT"; echo "release tree at $OUT"
