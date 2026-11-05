#!/bin/bash
#SBATCH --job-name=usg-llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SPLIT="${SPLIT:-dev}"
FEWSHOT="${FEWSHOT:-3}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# --- 1. Ada Modules & Environment -------------------------------------------
source /etc/profile.d/modules.sh 2>/dev/null || true
module load u22/python/3.12.4 2>/dev/null || true
module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/11.8 2>/dev/null || true

# --- 2. Cache Setup (Avoid Ada Home Quota Exceeded) -------------------------
ACTUAL_USER="${USER:-devshreevyas}"
if [ -d "/scratch" ] && mkdir -p "/scratch/${ACTUAL_USER}/hf_cache" 2>/dev/null; then
    export HF_HOME="/scratch/${ACTUAL_USER}/hf_cache"
    export TRANSFORMERS_CACHE="/scratch/${ACTUAL_USER}/hf_cache"
    export PIP_CACHE_DIR="/scratch/${ACTUAL_USER}/pip_cache"
    export TMPDIR="/scratch/${ACTUAL_USER}/pip_cache"
else
    mkdir -p "$HOME/.cache/hf_cache"
    export HF_HOME="$HOME/.cache/hf_cache"
fi

# Pick an open ephemeral port to avoid collisions on shared nodes
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "=========================================================="
echo " Starting USG NLP Extractor on Ada Cluster"
echo " Node: $(hostname) | Port: $PORT"
echo " Model: $MODEL | Split: $SPLIT"
echo " Cache Dir: $HF_HOME"
echo "=========================================================="

python3 -c "import vllm" 2>/dev/null || pip install --quiet "vllm>=0.6"
pip install --quiet -r requirements.txt

# --- 3. Start vLLM Model Server ---------------------------------------------
python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
    --max-model-len 16384 --dtype auto --gpu-memory-utilization 0.90 > "logs/vllm-${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "Waiting for vLLM to spin up..."
for _ in $(seq 1 180); do
    curl -sf "http://localhost:${PORT}/v1/models" > /dev/null && break
    sleep 5
done
curl -sf "http://localhost:${PORT}/v1/models" > /dev/null || { echo "vLLM did not start in time"; tail -50 "logs/vllm-${SLURM_JOB_ID:-local}.log"; exit 1; }
echo "vLLM is ready!"

# --- 4. Run Extraction & Evaluation ------------------------------------------
python3 run_pipeline.py --extractor llm --backend openai --base-url "http://localhost:${PORT}/v1" \
    --model "$MODEL" --split "$SPLIT" --few-shot "$FEWSHOT" --workers 8

python3 run_pipeline.py --compare
