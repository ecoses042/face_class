# config.py

## 목적

프로젝트 전체에서 공유하는 경로, JSON 필드명, 전처리 하이퍼파라미터를 한 곳에서 관리한다.

## 주요 설정

| 변수 | 설명 |
|------|------|
| `RAW_DIR` | AI-Hub 원본 zip 루트 경로 |
| `PROC_DIR` | 전처리 결과 저장 경로 |
| `FRONTAL_SYM_THRESH` | 정면 판별 대칭 비율 임계값 (기본 0.25) |
| `MIN_BOX_SIZE` | 유효 bbox 최소 픽셀 크기 (기본 50) |
| `MIN_AGE` / `MAX_AGE` | 유효 나이 범위 (1~80) |
| `RANDOM_SEED` | train/valid/test 분할 재현성 |

## JSON 필드명 상수

`FIELD_*` 상수는 AI-Hub JSON 라벨의 실제 키 이름이다.  
데이터 포맷이 바뀌면 이 파일만 수정하면 된다.
