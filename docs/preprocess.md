# preprocess.py

## 목적

AI-Hub 원본 zip 파일을 풀고 JSON 라벨을 파싱하여 `dataset/processed/metadata.csv`를 생성한다.

## 실행

```bash
# 라벨만 처리 (이미지 zip 해제 생략)
python src/preprocess.py --skip-images

# 라벨 + 이미지 전체 처리
python src/preprocess.py

# zip 해제 없이 JSON 파싱만 재실행
python src/preprocess.py --skip-extract
```

## 처리 흐름

```
TL_*.zip / VL_*.zip  →  dataset/processed/labels/{Training,Validation}/*.json
TS_*.zip / VS_*.zip  →  dataset/processed/images/{Training,Validation}/*.png

JSON 파싱
 ├─ bbox 최소 크기 필터 (MIN_BOX_SIZE)
 ├─ 나이 범위 필터 (MIN_AGE ~ MAX_AGE)
 ├─ 정면 판별 (5점 랜드마크 대칭 비율)
 └─ photo_age = age_now - age_past

→ dataset/processed/metadata.csv
```

## metadata.csv 컬럼

| 컬럼 | 설명 |
|------|------|
| `filename` | JSON/이미지 파일명 (확장자 제외) |
| `split` | training / validation |
| `person_id` | 인물 ID (train/test 분리 기준) |
| `birth` | 출생연도 |
| `age_now` | 데이터 수집 시점 나이 |
| `age_past` | 사진이 몇 년 전 촬영인지 |
| `photo_age` | 실제 촬영 시점 나이 (= age_now - age_past) ← **모델 정답값** |
| `gender` | male / female |
| `bbox_x/y/w/h` | 얼굴 bbox |
| `is_frontal` | 1=정면, 0=비정면 (랜드마크 대칭 기반) |
| `image_path` | 이미지 파일 절대 경로 (없으면 NaN) |
| `label_path` | JSON 파일 절대 경로 |

## 실행 결과 (2026-05-23 기준)

- 라벨 zip 해제: Training 803개(40,150 JSON), Validation 101개(5,050 JSON)
- 유효 레코드: 41,770개 (정면 39,102 / 비정면 2,668)
- photo_age: mean=17.4, range=1~79
