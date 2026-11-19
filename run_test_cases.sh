#!/usr/bin/env bash
#
# 执行 5 个 KV cache / 压测用例
#
# 使用前请：
#   1. 修改 config.py（MODEL_PATH / WORK_PATH / HOST_IP / HOST_PORT 等）
#   2. 生成数据集: bash run_test_cases.sh generate
#   3. 执行测试:   bash run_test_cases.sh run [case_id|all]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi

# 可通过环境变量覆盖
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/datasets/custom_cases}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_NUM_MULTIPLIER="${DATA_NUM_MULTIPLIER:-4}"
REQUEST_RATE="${REQUEST_RATE:-0}"
TEST_TYPE="${TEST_TYPE:-stream}"
NPU_NUM="${NPU_NUM:-1}"
SEED="${SEED:-42}"

CASE1_DATASET="${DATASET_DIR}/case1_block_boundary_kv_cache.jsonl"
CASE2_DATASET="${DATASET_DIR}/case2_max_len_boundary.jsonl"
CASE3_DATASET="${DATASET_DIR}/case3_business_traffic.jsonl"
CASE4_DATASET="${DATASET_DIR}/case4_long_output_32k_in.jsonl"
CASE5_DATASET="${DATASET_DIR}/case5_long_tail.jsonl"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

generate_datasets() {
  log "开始生成 GLM-5 测试数据集 -> ${DATASET_DIR}"
  log "tokenizer: ${MODEL_PATH:-config.py MODEL_PATH / TOKENIZER_FALLBACK}"
  log "max_model_len: ${MAX_MODEL_LEN}"
  mkdir -p "${DATASET_DIR}"
  GEN_ARGS=(--save_dir "${DATASET_DIR}" --max_model_len "${MAX_MODEL_LEN}" --data_num_multiplier "${DATA_NUM_MULTIPLIER}" --seed "${SEED}" --cases all)
  if [[ -n "${MODEL_PATH}" ]]; then
    GEN_ARGS+=(--model_path "${MODEL_PATH}")
  fi
  "${PYTHON}" generate_custom_datasets.py "${GEN_ARGS[@]}"
  log "数据集生成完成"
}

run_case1() {
  log "======== 用例1: block-size 边界 (128n/128n±1/128n±2) ========"
  log "并发=200, 输出=1k, 验证 KV cache block-size 边界传输可靠性"
  "${PYTHON}" aisbench_test.py \
    --dataset "${CASE1_DATASET}" \
    --output_len 1024 \
    --concurrency 200 \
    --request_rate "${REQUEST_RATE}" \
    --test_type "${TEST_TYPE}" \
    --npu_num "${NPU_NUM}"
}

run_case2() {
  log "======== 用例2: 最大长度边界 (max_model_len±3) ========"
  log "并发=100, 输出=1k, max_model_len=${MAX_MODEL_LEN}"
  "${PYTHON}" aisbench_test.py \
    --dataset "${CASE2_DATASET}" \
    --output_len 1024 \
    --concurrency 100 \
    --request_rate "${REQUEST_RATE}" \
    --test_type "${TEST_TYPE}" \
    --npu_num "${NPU_NUM}"
}

run_case3() {
  log "======== 用例3: 业务流量压测 (30k/64k/90k) ========"
  log "并发=200, 输出=1k, 占比 50%/40%/10%"
  "${PYTHON}" aisbench_test.py \
    --dataset "${CASE3_DATASET}" \
    --output_len 1024 \
    --concurrency 200 \
    --request_rate "${REQUEST_RATE}" \
    --test_type "${TEST_TYPE}" \
    --npu_num "${NPU_NUM}"
}

run_case4() {
  log "======== 用例4: 长输出重计算 (32k 输入 / 16k 输出) ========"
  log "并发=100"
  "${PYTHON}" aisbench_test.py \
    --dataset "${CASE4_DATASET}" \
    --output_len 16384 \
    --concurrency 100 \
    --request_rate "${REQUEST_RATE}" \
    --test_type "${TEST_TYPE}" \
    --npu_num "${NPU_NUM}"
}

run_case5() {
  log "======== 用例5: 长尾分布 (2k 90% / 128k 10%) ========"
  log "并发=200, 输出=1k"
  "${PYTHON}" aisbench_test.py \
    --dataset "${CASE5_DATASET}" \
    --output_len 1024 \
    --concurrency 200 \
    --request_rate "${REQUEST_RATE}" \
    --test_type "${TEST_TYPE}" \
    --npu_num "${NPU_NUM}"
}

run_all() {
  run_case1
  run_case2
  run_case3
  run_case4
  run_case5
  log "全部用例执行完成，结果见 aisbench_result.csv"
}

generate_prefix_mixed() {
  log "开始生成 prefix mixed 数据集 (10%/20%/30%/40% 命中率) -> ${DATASET_DIR}"
  log "tokenizer: ${MODEL_PATH:-config.py MODEL_PATH / TOKENIZER_FALLBACK}"
  mkdir -p "${DATASET_DIR}"
  GEN_ARGS=(--save_dir "${DATASET_DIR}" --cases prefix_mixed --prefix_data_num "${PREFIX_DATA_NUM:-800}" --seed "${SEED}" --prefix_num "${PREFIX_NUM:-1}" --dp "${DP:-1}")
  if [[ -n "${MODEL_PATH}" ]]; then
    GEN_ARGS+=(--model_path "${MODEL_PATH}")
  fi
  "${PYTHON}" generate_custom_datasets.py "${GEN_ARGS[@]}"
  log "prefix mixed 数据集生成完成"
}

usage() {
  cat <<EOF
用法:
  bash run_test_cases.sh generate          # 生成 5 个用例数据集
  bash run_test_cases.sh generate-prefix     # 生成 4 个 prefix mixed 命中率数据集
  bash run_test_cases.sh run [1|2|3|4|5|all]  # 执行指定用例（默认 all）

环境变量:
  DATASET_DIR          数据集目录 (默认: ./datasets/custom_cases)
  MAX_MODEL_LEN        用例2 最大模型长度 (默认: 131072)
  DATA_NUM_MULTIPLIER  数据集条数 = 并发 * 倍数 (默认: 4)
  REQUEST_RATE         请求频率 (默认: 0)
  TEST_TYPE            stream 或 text (默认: stream)
  NPU_NUM              NPU 卡数 (默认: 1)
  SEED                 随机种子 (默认: 42)
  PREFIX_DATA_NUM      prefix mixed 数据集条数 (默认: 800)
  PREFIX_NUM           prefix 种类数 (默认: 1)
  DP                   dp 域数量 (默认: 1)

示例:
  MAX_MODEL_LEN=65536 bash run_test_cases.sh generate
  bash run_test_cases.sh run 1
  bash run_test_cases.sh run all
EOF
}

main() {
  local action="${1:-}"
  local target="${2:-all}"

  case "${action}" in
    generate|gen)
      generate_datasets
      ;;
    generate-prefix|gen-prefix)
      generate_prefix_mixed
      ;;
    run)
      case "${target}" in
        1) run_case1 ;;
        2) run_case2 ;;
        3) run_case3 ;;
        4) run_case4 ;;
        5) run_case5 ;;
        all) run_all ;;
        *)
          echo "未知用例: ${target}"
          usage
          exit 1
          ;;
      esac
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
