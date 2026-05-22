#!/usr/bin/env bash
set -u -o pipefail

# Retry wrapper for single-GPU resumable eval.
# Behavior:
#   - Exit code 0: success, stop.
#   - Non-zero: retry after SLEEP_SECONDS until MAX_RETRIES reached.
#   - OOM is still detected and logged explicitly.
#
# Why retry on any non-zero?
#   Some failures (e.g., exit 139) may not print a Python traceback, so treating
#   only explicit OOM as retriable can stop progress unexpectedly.

MAX_RETRIES="${MAX_RETRIES:-10}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_eval_resume_single_gpu.sh}"
LOG_DIR_LOCAL="${LOG_DIR_LOCAL:-./logs}"
mkdir -p "${LOG_DIR_LOCAL}"

# Optional: keep legacy behavior by setting RETRY_ON_ANY_NONZERO=0.
RETRY_ON_ANY_NONZERO="${RETRY_ON_ANY_NONZERO:-1}"

attempt=1
run_ts="$(date +%Y%m%d_%H%M%S)"

echo "[INFO] retry wrapper started"
echo "[INFO] RUN_SCRIPT=${RUN_SCRIPT}"
echo "[INFO] MAX_RETRIES=${MAX_RETRIES}, SLEEP_SECONDS=${SLEEP_SECONDS}, RETRY_ON_ANY_NONZERO=${RETRY_ON_ANY_NONZERO}"

while (( attempt <= MAX_RETRIES )); do
  log_path="${LOG_DIR_LOCAL}/eval_retry_attempt_${attempt}_${run_ts}.log"
  echo "============================================================"
  echo "[INFO] Attempt ${attempt}/${MAX_RETRIES} at $(date)"
  echo "[INFO] Logging to ${log_path}"
  echo "============================================================"

  bash "${RUN_SCRIPT}" 2>&1 | tee "${log_path}"
  exit_code=${PIPESTATUS[0]}

  if [[ ${exit_code} -eq 0 ]]; then
    echo "[SUCCESS] Eval completed successfully on attempt ${attempt}."
    exit 0
  fi

  is_oom=0
  if grep -Eqi "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError:.*out of memory" "${log_path}"; then
    is_oom=1
    echo "[WARN] CUDA OOM detected (exit=${exit_code})."
  else
    echo "[WARN] Non-zero exit without explicit OOM signature (exit=${exit_code})."
  fi

  if [[ "${RETRY_ON_ANY_NONZERO}" != "1" && ${is_oom} -ne 1 ]]; then
    echo "[FAIL] RETRY_ON_ANY_NONZERO=0 and this failure is non-OOM. Stopping."
    exit "${exit_code}"
  fi

  if (( attempt >= MAX_RETRIES )); then
    echo "[FAIL] Reached MAX_RETRIES=${MAX_RETRIES}. Last exit=${exit_code}."
    exit "${exit_code}"
  fi

  echo "[INFO] Sleeping ${SLEEP_SECONDS}s before retry..."
  sleep "${SLEEP_SECONDS}"
  ((attempt++))
done

echo "[FAIL] Unexpected termination."
exit 99
