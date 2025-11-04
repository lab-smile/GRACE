#!/usr/bin/env bash
set -euo pipefail
echo "[DEBUG] Entrypoint script has started."
usage() {
  cat <<'USAGE'
GRACE Container Entrypoint

Stages:
  preprocess  → runs your /workspace/preprocess.py
  train       → runs your /workspace/train.py
  test        → runs your /workspace/test.py

Defaults (override via env or flags):
  Data directory (read & write):   $GRACE_DATA_DIR

Usage:
  preprocess [--data_dir DIR] [--source-folders FOLDER [FOLDER ...]] [--verbose]
  train      [--data_dir DIR]  [other GRACE train flags...]
  test       [--data_dir DIR]  [other GRACE test  flags...]

Examples:
  preprocess --data_dir /data --source-folders folder1 folder2 folder3 --verbose

  train --data_dir /data --num_gpu 1 --model_save_name grace --max_iteration 1000 --spatial_size 64 --json_name dataset_1.json --a_min_value 0 --a_max_value 255 --N_classes 12

  test  --data_dir /data --num_gpu 1 --model_load_name grace.pth --spatial_size 64 --a_min_value 0 --a_max_value 255 --N_classes 12 --data_parallel False --batch_size_test 1
USAGE
}

stage="${1:-}"
if [[ -z "${stage}" || "${stage}" == "--help" || "${stage}" == "-h" ]]; then
  usage; exit 0
fi
shift

DATA_DIR="${GRACE_DATA_DIR:-/data}"
PREPROCESS_SCRIPT="${GRACE_PREPROCESS_SCRIPT:-/workspace/preprocess.py}"
TRAIN_SCRIPT="${GRACE_TRAIN_SCRIPT:-/workspace/train.py}"
TEST_SCRIPT="${GRACE_TEST_SCRIPT:-/workspace/test.py}"

run_as_owner() {
  local -a CMD=( "$@" )
  if owner_u=$(stat -c '%u' "$DATA_DIR" 2>/dev/null) && owner_g=$(stat -c '%g' "$DATA_DIR" 2>/dev/null); then
    if [[ "$owner_u" -ne 0 || "$owner_g" -ne 0 ]]; then
      getent group "$owner_g" >/dev/null 2>&1 || groupadd -g "$owner_g" hostgroup || true
      id -u "$owner_u" >/dev/null 2>&1 || useradd -m -u "$owner_u" -g "$owner_g" hostuser || true
      umask "${UMASK:-0002}"
      gosu "$owner_u:$owner_g" "${CMD[@]}"
      return $?
    fi
  fi
  exec "${CMD[@]}" 

  return $?
}

case "${stage}" in
  preprocess)
    INPUT="${DATA_DIR}"
    FOLDERS=()
    VERBOSE=false
    # Parse flags
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --data_dir)       		INPUT="$2"; shift 2;;
        --source-folders|--source_folders)
          shift
            while [[ $# -gt 0 && $1 != --* ]]; do
            FOLDERS+=("$1")
                shift
            done
            ;;
        --verbose)		VERBOSE=true; shift;;		
              *) echo "Unknown arg for preprocess: $1"; usage; exit 1;;
      esac
    done
    CMD=(python "${PREPROCESS_SCRIPT}" --data "${INPUT}")
    if [[ ${#FOLDERS[@]} -gt 0 ]]; then
	    CMD+=(--source-folders "${FOLDERS[@]}")
    fi
    if $VERBOSE; then
	    CMD+=(--verbose)
    fi

    echo "[GRACE] Preprocessing Data: ${CMD[*]}"; 
    run_as_owner "${CMD[@]}";;

  train)
    TRAIN_DATA="${DATA_DIR}"
    GPUS=1
    SNAME="grace"
    BATCH_SIZE_TRAIN=1
    BATCH_SIZE_VAL=1
    MAX_ITER=100
    SPATIAL_SIZE=64
    DATASET="dataset_1.json"
    A_MIN_VAL=0
    A_MAX_VAL=255
    N_CLASSES=12
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --data_dir)             TRAIN_DATA="$2"; TRAIN_OUT="$2"; shift 2;;
        --num_gpu)              GPUS="$2"; shift 2;;
        --model_save_name)      SNAME="$2"; shift 2;;
        --batch_size_train)     BATCH_SIZE_TRAIN="$2"; shift 2;;
        --batch_size_validation) BATCH_SIZE_VAL="$2"; shift 2;;
        --max_iteration)       MAX_ITER="$2"; shift 2;;
        --spatial_size)         SPATIAL_SIZE="$2"; shift 2;;
        --json_name)         DATASET="$2"; shift 2;;
        --a_min_value)         A_MIN_VAL="$2"; shift 2;;
        --a_max_value)         A_MAX_VAL="$2"; shift 2;;
        --N_classes)         N_CLASSES="$2"; shift 2;;
        *) echo "Unknown arg for train: $1"; usage; exit 1;;
      esac
    done

    if [[ "${TRAIN_DATA: -1}" != "/" ]]; then
	    TRAIN_DATA="${TRAIN_DATA}/"
    fi
    CMD=( python "${TRAIN_SCRIPT}" --data_dir "${TRAIN_DATA}" --num_gpu "${GPUS}" \
          --model_save_name "${SNAME}" --batch_size_train "${BATCH_SIZE_TRAIN}" \
          --batch_size_validation "${BATCH_SIZE_VAL}" --max_iteration "${MAX_ITER}" \
          --spatial_size "${SPATIAL_SIZE}" --json_name "${DATASET}" --a_min_value "${A_MIN_VAL}" \
          --a_max_value "${A_MAX_VAL}" --N_classes "${N_CLASSES}" )
    echo "[GRACE] Training: ${CMD[*]}";
    run_as_owner "${CMD[@]}";;


  test)
    TEST_DATA="${CURRENT_DIR}"
    GPUS=1
    LNAME="grace"
    SPATIAL_SIZE=64
    A_MIN_VAL=0
    A_MAX_VAL=255
    N_CLASSES=12
    DATA_PARALLEL="False"
    BATCH_SIZE_TEST=1
    DATASET="dataset_1.json"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --data_dir)             TEST_DATA="$2"; TEST_OUT="$2"; shift 2;;
        --num_gpu)              GPUS="$2"; shift 2;;
        --model_load_name)      LNAME="$2"; shift 2;;
        --spatial_size)         SPATIAL_SIZE="$2"; shift 2;;
        --a_min_value)         A_MIN_VAL="$2"; shift 2;;
        --a_max_value)         A_MAX_VAL="$2"; shift 2;;
        --N_classes)         N_CLASSES="$2"; shift 2;;
        --data_parallel)        DATA_PARALLEL="$2"; shift 2;;
        --batch_size_test)      BATCH_SIZE_TEST="$2"; shift 2;;
        --json_name)         DATASET="$2"; shift 2;;
        *) echo "Unknown arg for test: $1"; usage; exit 1;;
      esac
    done
    if [[ "${TEST_DATA: -1}" != "/" ]]; then
	    TEST_DATA="${TEST_DATA}/"
    fi
    CMD=( python "${TEST_SCRIPT}" --data_dir "${TEST_DATA}" --num_gpu "${GPUS}" \
          --model_load_name "${LNAME}" --spatial_size "${SPATIAL_SIZE}"  \
          --a_min_value "${A_MIN_VAL}" --a_max_value "${A_MAX_VAL}" --N_classes "${N_CLASSES}" \
          --data_parallel "${DATA_PARALLEL}" --batch_size_test "${BATCH_SIZE_TEST}" 
          --json_name "${DATASET}" )
    echo "[GRACE] Testing: ${CMD[*]}";
    run_as_owner "${CMD[@]}";;

  *)
    echo "Unknown stage: ${stage}"
    usage
    exit 1
    ;;
esac


