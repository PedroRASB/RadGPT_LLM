#!/usr/bin/env bash
# Usage:
#   bash LaunchMultiGPUFlex.sh [DATA_PATH] [SAVE_NAME] [STEP] [LLM_SIZE] [NUM_GPUS] [INST_PER_GPU] [BASE_GPU] [TOP_GPU_USAGE] [HF_CACHE]
#
# Examples:
#   1) Use defaults except for data/save paths:
#      bash LaunchMultiGPUFlex.sh /my_data.feather /my_save.csv
#
#   2) Override everything (9 arguments):
#      bash LaunchMultiGPUFlex.sh \
#         reports_concat.feather \
#         reports_concat_size_and_type.csv \
#         "type and size multi-organ" \
#         large \
#         4 \
#         1 \
#         0 \
#         0.75 \
#         /mnt/sdh/pedro/HFCache
#
#   3) Only override HF_CACHE (argument #9), keep others default:
#      bash LaunchMultiGPUFlex.sh /my_data.feather /my_save.csv "" "" "" "" "" "" /mnt/sdh/pedro/HFCache
#
# Notes:
#   - If you pass "" (empty string) for an argument, it reverts to its default.
#   - The 8th argument is TOP_GPU_USAGE, default 0.95 (max fraction of GPU memory usage if 1 instance/GPU).
#   - The 9th argument is HF_CACHE, default ./HFCache (where model is downloaded).
#
# Before running, activate any necessary conda environment, for example:
#   source /path/to/anaconda3/etc/profile.d/conda.sh
#   conda activate vllm2


##############################################################################
# 1) Exit on any error
##############################################################################
set -euxo pipefail

##############################################################################
# 2) Parse arguments with defaults
##############################################################################
DATA_PATH="${1:-/path/to/data.feather}"   # 1) Path to your dataset
SAVE_NAME="${2:-/path/to/save.csv}"       # 2) Path/filename to save results
STEP="${3:-diagnoses}"                    # 3) A string for the Python code's --step
LLM_SIZE="${4:-small}"                    # 4) "small" or "large" model
NUM_GPUS="${5:-8}"                        # 5) Number of GPUs to use
INST_PER_GPU="${6:-1}"                    # 6) Number of VLLM instances per GPU
BASE_GPU="${7:-0}"                        # 7) Base GPU index
TOP_GPU_USAGE="${8:-0.95}"                # 8) The top fraction of GPU memory usage (default 0.95)
HF_CACHE="${9:-./HFCache}"                # 9) The HF cache directory (default ./HFCache)

##############################################################################
# 3) Compute total instances
##############################################################################
TOTAL_INSTANCES=$((NUM_GPUS * INST_PER_GPU))

##############################################################################
# 4) Randomly pick a BASE_PORT, ensuring subsequent ports are free
##############################################################################
randomize_base_port() {
  local start_range=1000
  local end_range=9999

  while true; do
    # Pick a random port in [start_range..end_range]
    try_port=$((start_range + RANDOM % (end_range - start_range + 1)))

    # Check if all needed ports [try_port..(try_port + TOTAL_INSTANCES - 1)] are free
    all_free=true
    for offset in $(seq 0 $((TOTAL_INSTANCES - 1))); do
      p=$((try_port + offset))
      # If port is taken, try again
      if lsof -i :"$p" -sTCP:LISTEN > /dev/null 2>&1; then
        all_free=false
        break
      fi
    done

    if $all_free; then
      echo "$try_port"
      return 0
    fi
  done
}

BASE_PORT=$(randomize_base_port)
echo "Selected BASE_PORT=$BASE_PORT"

##############################################################################
# 5) Decide Which Model/Arguments to Use (small vs. large)
##############################################################################
case "$LLM_SIZE" in
  small)
    # The ~8B GPTQ model
    MODEL="iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8"
    MODEL_OPTS="--max-model-len 12000 --dtype float16"
    ;;
  large)
    # The 70B AWQ-INT4 model
    MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    MODEL_OPTS="--dtype half --tensor-parallel-size 1 --max-model-len 60000"
    ;;
  *)
    echo "Unknown LLM_SIZE: '$LLM_SIZE'. Must be 'small' or 'large'."
    exit 1
    ;;
esac

##############################################################################
# 6) GPU Memory Utilization Based on INST_PER_GPU & TOP_GPU_USAGE
##############################################################################
case "$INST_PER_GPU" in
  1)
    GPU_MEM="$TOP_GPU_USAGE"
    ;;
  2)
    GPU_MEM="$(python -c "print(round($TOP_GPU_USAGE / 2, 2))")"
    ;;
  3)
    GPU_MEM="$(python -c "print(round($TOP_GPU_USAGE / 3, 2))")"
    ;;
  *)
    GPU_MEM="$(python -c "print(round($TOP_GPU_USAGE / $INST_PER_GPU, 2))")"
    ;;
esac

##############################################################################
# 7) Launch VLLM Instances
##############################################################################
instance_id=0
for gpu_index in $(seq 0 $((NUM_GPUS - 1))); do
  for ins_index in $(seq 0 $((INST_PER_GPU - 1))); do

    PORT=$((BASE_PORT + instance_id))
    GPU=$((BASE_GPU + gpu_index))

    echo "Launching VLLM instance #$instance_id on GPU $GPU (port $PORT)"
    echo "Memory per instance: $GPU_MEM"
    echo "HF_CACHE: $HF_CACHE"

    TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" CUDA_VISIBLE_DEVICES="$GPU" \
      vllm serve "$MODEL" \
                 $MODEL_OPTS \
                 --port "$PORT" \
                 --gpu_memory_utilization "$GPU_MEM" \
                 --enforce-eager \
                 > "API_GPU${GPU}_INS${ins_index}.log" 2>&1 &

    instance_id=$((instance_id + 1))
  done
done

##############################################################################
# 8) Wait for All VLLM Instances to be Ready
##############################################################################
for i in $(seq 0 $((TOTAL_INSTANCES - 1))); do
  PORT=$((BASE_PORT + i))
  while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null; do
    echo "Waiting for API on port $PORT..."
    sleep 5
  done
done

echo "All vllm APIs are ready. Running python scripts..." >> FastDiseases.log

##############################################################################
# 9) Launch Python Jobs (One per VLLM Instance)
##############################################################################
for i in $(seq 0 $((TOTAL_INSTANCES - 1))); do
  PORT=$((BASE_PORT + i))
  echo "Launching Python script for instance #$i on port $PORT"

  python RunRadGPT.py \
    --port "$PORT" \
    --data_path "$DATA_PATH" \
    --institution "UCSF" \
    --step "$STEP" \
    --save_name "$SAVE_NAME" \
    --parts "$TOTAL_INSTANCES" \
    --part "$i" \
    >> "LLM_part_${i}.log" 2>&1 &
done

##############################################################################
# 10) Wait for All Background Jobs
##############################################################################
wait