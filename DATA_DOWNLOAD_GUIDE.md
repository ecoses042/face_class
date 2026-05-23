# AI-Hub 데이터 다운로드 가이드

## 대상 데이터셋

**안면 인식 에이징(aging) 이미지 데이터**  
- AI-Hub 데이터셋 번호: 71415  
- 구축 연도: 2022년 / 최종 개방: 2023년 11월  
- 신청 자격: 내국인 (개인/기관)

---

## Step 1. AI-Hub 회원가입 및 로그인

1. https://aihub.or.kr 접속
2. 회원가입 (본인인증 필요)
3. 로그인

---

## Step 2. 데이터 이용신청

1. 검색창에 **"안면 인식 에이징"** 검색
2. 데이터 상세 페이지 진입
3. **[이용신청]** 버튼 클릭
4. 신청 목적 입력 후 제출
5. 승인 이메일 수신 확인 (즉시 또는 수일 소요)

---

## Step 3-A. 웹 다운로드 방식 (추천: 처음 시도하는 경우)

승인 후 데이터 상세 페이지 → **[다운로드]** 탭  
- 원하는 파일/파티션 선택 후 다운로드
- 저장 위치: `dataset/raw/aihub_aging/`

압축 해제:

```bash
cd /home/ecoses042/26-1/deeplearning/face_age/dataset/raw/aihub_aging
unzip "*.zip"      # zip 형식인 경우
# 또는
tar -xf "*.tar"    # tar 형식인 경우
```

---

## Step 3-B. aihubshell 다운로드 방식 (WSL 환경 추천)

### aihubshell 설치

```bash
# AI-Hub 다운로드 프로그램 설치 페이지에서 Ubuntu용 에이전트 다운로드
# https://www.aihub.or.kr/dwldPrgmInstall.do

# 설치 예시 (버전명은 실제 파일명으로 변경)
chmod +x aihubshell
sudo mv aihubshell /usr/local/bin/
aihubshell --version
```

### 데이터 다운로드

```bash
mkdir -p dataset/raw/aihub_aging
cd dataset/raw/aihub_aging

aihubshell -mode d \
  -datasetkey 71415 \
  -aihubid <AI_HUB_이메일> \
  -aihubpw <AI_HUB_비밀번호> \
  -savepath .
```

> **주의**: 실제 옵션명은 설치한 aihubshell 버전의 `--help` 출력을 기준으로 확인하세요.

---

## Step 4. 압축 해제 후 구조 확인

```bash
# 프로젝트 루트에서 실행
cd /home/ecoses042/26-1/deeplearning/face_age

# tree가 없으면: sudo apt install tree
tree -L 4 dataset/raw/aihub_aging

# 또는 진단 스크립트 실행 (JSON 샘플까지 출력)
python src/check_data_structure.py
```

---

## Step 5. 확인해야 할 항목

데이터를 받은 직후 아래 항목을 반드시 확인한다.

| 항목 | 확인 방법 |
|------|----------|
| 이미지 확장자 | `check_data_structure.py` 출력 |
| 라벨 파일 형식 (json/xml/csv) | `check_data_structure.py` 출력 |
| 나이(age) 필드명 | JSON 샘플 출력 확인 |
| 인물 ID 필드명 | JSON 샘플 출력 확인 |
| 얼굴 bbox 필드 | JSON 샘플 출력 확인 |
| 정면 여부 필드 | JSON 샘플 출력 확인 |
| train/validation 이미 분리 여부 | 디렉토리 구조 확인 |

---

## Step 6. 다음 단계

구조 확인 후 → `src/preprocess.py` 작성  
목표: `dataset/processed/metadata.csv` 생성

```csv
image_path,person_id,age,gender,bbox,is_frontal,split
dataset/processed/images/000001.jpg,P001,23,F,"[x,y,w,h]",1,train
```

---

## 데이터 이용 명시 문구

논문/보고서에 다음 문구를 포함해야 합니다.

> 본 연구는 AI-Hub에서 제공하는 "안면 인식 에이징(aging) 이미지 데이터"를 활용하여 수행되었다.
