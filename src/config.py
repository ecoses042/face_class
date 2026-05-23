from pathlib import Path

# --- 경로 ---
ROOT = Path(__file__).parent.parent

RAW_DIR     = ROOT / "dataset/raw/aihub_aging/118.안면_인식_에이징(aging)_이미지_데이터/01-1.정식개방데이터"
PROC_DIR    = ROOT / "dataset/processed"
PROC_IMAGES = PROC_DIR / "images"
PROC_LABELS = PROC_DIR / "labels"
EMBED_DIR   = ROOT / "dataset/embeddings"
RESULTS_DIR = ROOT / "results"

# --- 원천/라벨 zip 폴더명 ---
IMG_SUBDIR   = "01.원천데이터"
LABEL_SUBDIR = "02.라벨링데이터"

# --- JSON 필드명 ---
FIELD_ID       = "id"
FIELD_BIRTH    = "birth"
FIELD_AGE_NOW  = "age_now"
FIELD_AGE_PAST = "age_past"
FIELD_GENDER   = "gender"
FIELD_FORMAT   = "format"
FIELD_ANNOT    = "annotation"
FIELD_BOX      = "box"
FIELD_LANDMARK = "landmark"

# --- 전처리 설정 ---
FRONTAL_SYM_THRESH = 0.25   # 정면 판별 대칭 비율 임계값 (낮을수록 엄격)
MIN_BOX_SIZE       = 50     # bbox 최소 너비/높이 (픽셀)
MIN_AGE            = 1      # 유효 나이 범위
MAX_AGE            = 80

# --- 데이터 분할 ---
VALID_RATIO = 0.1
TEST_RATIO  = 0.1
RANDOM_SEED = 42
