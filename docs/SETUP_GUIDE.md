# SETUP_GUIDE.md 문서 설명

## 목적

`SETUP_GUIDE.md`는 저장소를 처음 받은 사람이 환경 구성부터 모델 실행까지 완료할 수 있도록 안내하는 셋업 가이드입니다.

## 대상 독자

- 이 프로젝트를 처음 clone하는 사람
- 새 환경에서 git pull 후 설정하는 사람

## 주요 내용

| 섹션 | 내용 |
|------|------|
| 1. 저장소 클론 | git clone 명령 |
| 2. 가상환경 설치 | venv 생성 및 requirements.txt 설치 |
| 3. 모델 다운로드 | Hugging Face Hub (ecoses042/face-age-estimation)에서 모델 5개 다운로드 |
| 4. 추론 실행 | predict.py, predict_all.py 사용법 |
| 5. 성능 요약 | 모델별 MAE 비교표 |
| 6. 재학습 | train_fcn.py 등 학습 스크립트 사용법 |
| 문제 해결 | 자주 발생하는 오류와 해결 방법 |

## 관련 파일

- `SETUP_GUIDE.md` — 실제 가이드 문서
- `requirements.txt` — 패키지 목록
- `src/predict.py` — 단일 이미지 예측
- `src/predict_all.py` — 다중 이미지 일괄 예측
- `docs/USAGE_GUIDE.md` — 상세 사용법
