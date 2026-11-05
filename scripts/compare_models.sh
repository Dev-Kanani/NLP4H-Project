#!/bin/bash
# Laptop/workstation comparison with Ollama: rule baseline + three open-weight model families
# (Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B v0.3) on the same split, then a comparison table.
#
#   bash scripts/compare_models.sh            # dev split
#
# On a GPU cluster use scripts/ada_vllm.sbatch once per model instead (vLLM serves one model per server):
#   for m in Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3; do
#       MODEL=$m sbatch scripts/ada_vllm.sbatch; done
#
# Every model response is cached in outputs/llm_cache/, so a run can be reproduced without the model:
#   python run_pipeline.py --extractor llm --backend replay --model <same model name>
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT="${SPLIT:-dev}"
MODELS=("qwen2.5:7b-instruct" "llama3.1:8b-instruct-q8_0" "mistral:7b-instruct-v0.3-q8_0")

python run_pipeline.py --split "$SPLIT"   # rule baseline
for m in "${MODELS[@]}"; do
    ollama pull "$m"
    python run_pipeline.py --extractor llm --backend ollama --model "$m" --split "$SPLIT"
done
python run_pipeline.py --compare
