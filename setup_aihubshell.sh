#!/usr/bin/env bash
# aihubshell 설치 스크립트 (Linux / Ubuntu / WSL 공통)
set -e

INSTALL_DIR="/usr/local/bin"
SHELL_NAME="aihubshell"
DOWNLOAD_URL="https://api.aihub.or.kr/api/aihubshell.do"

echo "=== aihubshell 설치 시작 ==="

# 다운로드
echo "[1/3] 다운로드 중..."
curl -o "$SHELL_NAME" "$DOWNLOAD_URL"

# 실행 권한
echo "[2/3] 실행 권한 부여..."
chmod +x "$SHELL_NAME"

# 전역 등록
echo "[3/3] $INSTALL_DIR 에 복사 (sudo 필요)..."
sudo cp "$SHELL_NAME" "$INSTALL_DIR/"
rm "$SHELL_NAME"

echo ""
echo "=== 설치 완료 ==="
aihubshell --version 2>/dev/null || aihubshell -version 2>/dev/null || echo "(버전 확인: aihubshell 실행 후 옵션 없이 실행해 보세요)"
echo ""
echo "다음 단계:"
echo "  1. AI-Hub 마이페이지에서 API Key 발급"
echo "  2. 데이터셋 활용신청 승인 확인"
echo "  3. 아래 명령어로 다운로드:"
echo ""
echo "  aihubshell -mode d -datasetkey 71415 -aihubapikey 'YOUR_API_KEY'"
