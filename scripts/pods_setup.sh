#!/usr/bin/env bash
# Prepare a fresh RunPod pod. Run from the repo root after cloning:
#   bash scripts/pod_setup.sh             # Qwen / main environment
#   bash scripts/pod_setup.sh --deepseek  # also build the DeepSeek-VL2 env (see DEEPSEEK.md)
# Safe to re-run. /workspace (network volume) keeps data, runs and the model cache.
set -euo pipefail

getent hosts github.com >/dev/null || { echo "DNS broken: run  echo 'nameserver 8.8.8.8' > /etc/resolv.conf"; exit 1; }

for d in data raw runs; do
  mkdir -p /workspace/sp/$d
  [ -e "$d" ] || ln -s /workspace/sp/$d "$d"
done

grep -q 'HF_HOME=/workspace/hf' ~/.bashrc || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

pip install -q "transformers>=4.49" accelerate peft huggingface_hub pillow pytest
python -m pytest -q
if [ "${1:-}" = "--deepseek" ]; then
  # Separate Python 3.10 env: DeepSeek-VL2 pins torch==2.0.1 and transformers==4.38.2.
  pip install -q uv
  [ -d /root/venv-ds ] || uv venv --python 3.10 /root/venv-ds
  [ -d /root/DeepSeek-VL2 ] || git clone https://github.com/deepseek-ai/DeepSeek-VL2.git /root/DeepSeek-VL2
  export UV_CACHE_DIR=/root/.cache/uv  # uv's cache must not live on the network volume either
  VIRTUAL_ENV=/root/venv-ds uv pip install -e /root/DeepSeek-VL2 "xformers==0.0.21" "numpy<2" hf_transfer pillow
  echo "DeepSeek env ready. Activate with: source /root/venv-ds/bin/activate"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Ready. Run: source ~/.bashrc"
