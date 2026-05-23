# aihubshell 설치 및 사용 가이드

출처: https://aihub.or.kr/devsport/apishell/list.do  
현재 버전: `aihubshell version 25.09.19 v0.6` (Bash 스크립트 형태)

---

## 1. 설치

```bash
# 스크립트로 한 번에 설치
bash setup_aihubshell.sh

# 또는 수동 설치
curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do
chmod +x aihubshell
sudo cp aihubshell /usr/local/bin/
```

---

## 2. API Key 발급

1. https://aihub.or.kr 로그인
2. **마이페이지 → API key 발급** 버튼 클릭
3. 가입 이메일로 API key 수신

---

## 3. 데이터 다운로드 (안면 인식 에이징 데이터, key: 71415)

```bash
# 저장 위치로 이동
mkdir -p dataset/raw/aihub_aging
cd dataset/raw/aihub_aging

# 전체 다운로드
aihubshell -mode d \
  -datasetkey 71415 \
  -aihubapikey 'YOUR_API_KEY'

# 특정 파일만 선택하려면 (-filekey로 파일 번호 지정)
aihubshell -mode d \
  -datasetkey 71415 \
  -filekey 51937,51939 \
  -aihubapikey 'YOUR_API_KEY'
```

> **주의**: API key에 특수문자(`!`, `@`, `$`)가 있으면 반드시 **홑따옴표**로 감싸야 합니다.

---

## 4. 주요 명령어 정리

| 모드 | 기능 | 명령어 |
|------|------|--------|
| `-mode l` | 내 데이터셋 목록 조회 | `aihubshell -mode l` |
| `-mode d` | 데이터셋 다운로드 | `aihubshell -mode d -datasetkey 71415 -aihubapikey 'KEY'` |
| `-mode pl` | 데이터 패키지 목록 조회 | `aihubshell -mode pl -datapckagekey {번호}` |
| `-mode pd` | 데이터 패키지 다운로드 | `aihubshell -mode pd -datapckagekey {번호} -aihubapikey 'KEY'` |

---

## 5. 주의사항

- 데이터셋 **활용신청 승인** 완료 후에만 다운로드 가능
- 다운로드 데이터 용량의 **2~3배** 이상 디스크 여유 공간 필요
- WSL 환경에서 Linux와 동일하게 사용 가능
- 공식 가이드 PDF: https://aihub.or.kr/static/pdf/aihubshell_가이드.pdf

---

## 6. 다운로드 후 다음 단계

```bash
# 프로젝트 루트로 돌아와서 구조 확인
cd /home/ecoses042/26-1/deeplearning/face_age
python src/check_data_structure.py
```
