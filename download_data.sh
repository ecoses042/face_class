#!/usr/bin/env bash
# 라벨 전체(TL+VL) + Validation 이미지(VS) 다운로드
# 총 1005개 파일, 약 8 GB
# aihubshell은 현재 디렉토리에 다운로드하므로 cd 후 실행
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_DIR="$SCRIPT_DIR/dataset/raw/aihub_aging"
AIHUBSHELL="$SCRIPT_DIR/aihubshell"
API_KEY="9571FD02-4F03-4F0B-81B2-BCF931C48571"
DATASET_KEY="71415"
LOG_FILE="$SCRIPT_DIR/download.log"

mkdir -p "$SAVE_DIR"

echo "=== AI-Hub 데이터 다운로드 시작 ===" | tee -a "$LOG_FILE"
echo "저장 위치: $SAVE_DIR" | tee -a "$LOG_FILE"
echo "시작 시각: $(date)" | tee -a "$LOG_FILE"

BATCHES=(
    "/tmp/batch_01.txt"
    "/tmp/batch_02.txt"
    "/tmp/batch_03.txt"
    "/tmp/batch_04.txt"
    "/tmp/batch_05.txt"
    "/tmp/batch_06.txt"
)

TOTAL=${#BATCHES[@]}

# aihubshell은 savepath 옵션 없이 현재 디렉토리에 다운로드함
cd "$SAVE_DIR"

for i in "${!BATCHES[@]}"; do
    BATCH_FILE="${BATCHES[$i]}"
    BATCH_NUM=$((i + 1))
    FILEKEYS=$(cat "$BATCH_FILE")

    echo "" | tee -a "$LOG_FILE"
    echo "--- 배치 $BATCH_NUM / $TOTAL 시작: $(date) ---" | tee -a "$LOG_FILE"

    "$AIHUBSHELL" -mode d \
        -datasetkey "$DATASET_KEY" \
        -filekey "$FILEKEYS" \
        -aihubapikey "$API_KEY" 2>&1 | tee -a "$LOG_FILE"

    echo "--- 배치 $BATCH_NUM 완료: $(date) ---" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== 전체 다운로드 완료: $(date) ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "다음 단계: cd $SCRIPT_DIR && python src/check_data_structure.py"
