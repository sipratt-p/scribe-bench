#!/usr/bin/env bash
# Build an anonymized release tree: code, prompts, lexicon, splits, cached outputs (small), wiki, paper; no machine names or tailnet addresses.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/scribe-bench-release}
rm -rf "$OUT"; mkdir -p "$OUT"
rsync -a --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'runs/*.wav' --exclude 'runs/bundle' --exclude 'runs/pod' --exclude 'runs/pod_nano' --exclude 'runs/*.safetensors' --exclude 'runs/gemma_server.log' --exclude 'runs/**/*.log' --exclude 'data/primock57' --exclude 'data/aci-bench' --exclude 'data/MTS-Dialog' --exclude 'data/medsynth' --exclude 'data/sft' --exclude 'runs/expert_iter/adapter_*' --exclude 'runs/expert_iter/samples_*' --exclude 'runs/live_synth/lookups' --exclude 'demo/uploads' --exclude 'demo/data' --exclude 'runs/*.log' --exclude 'runs/*.png' --exclude 'wiki/abridge.md' --exclude 'wiki/interview-narrative.md' --exclude 'wiki/nvidia-engagement.md' --exclude 'release' ./ "$OUT/"
# scrub identifiers
grep -rl -E '100\.83\.231\.108|100\.90\.251\.52|beast|cosmo|sethcosmo|sipratt|tail2f894f|Seth|abridge' "$OUT" --include='*.py' --include='*.md' --include='*.html' --include='*.sh' --include='*.json' --include='*.toml' 2>/dev/null | while read -r f; do
  sed -i '' -e 's/100\.83\.231\.108/GPU_HOST/g' -e 's/100\.90\.251\.52/DEMO_HOST/g' -e 's/seth-cosmo-studio\.tail2f894f\.ts\.net/DEMO_HOST/g' -e 's/sethcosmo/user/g' -e 's/sipratt@gmail\.com/author@example.org/g' -e 's/Seth Pratt/Anonymous/g' -e "s/Seth's/the reviewer's/g" -e 's/Seth:/Reviewer:/g' -e 's/Seth\b/the reviewer/g' -e 's/beast-ubn/gpubox/g' -e 's/\bbeast\b/gpubox/g' -e 's/\bcosmo\b/workstation/g' -e 's/"abridge",\{0,1\}//g' -e 's/"interview-narrative",\{0,1\}//g' -e 's/"nvidia-engagement",\{0,1\}//g' "$f"
done
cat > "$OUT/data/README.md" <<'EOF'
Datasets are not redistributed here. Recreate this folder with the commands in the top-level README (PriMock57, ACI-Bench, MTS-Dialog, MedSynth clones; `mixed/` built with sox; `primock_export/` built by `scribe_bench.primock`). `lexicon.txt`, the NHS A-Z and patient.info condition indexes and `primock_export/` are included because they are derived files the experiments read directly.
EOF
echo "remaining identifiers:"; grep -rn -E '100\.83\.231\.108|100\.90\.251\.52|sethcosmo|sipratt|tail2f894f' "$OUT" | head -n 5 || true
du -sh "$OUT"; echo "release tree at $OUT"
