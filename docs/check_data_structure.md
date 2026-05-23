# check_data_structure.py

## 목적

`dataset/raw/aihub_aging/` 아래에 AI-Hub 데이터를 내려받은 뒤 바로 실행하는 진단 스크립트.  
디렉토리 트리, 확장자별 파일 수, JSON 라벨 샘플을 출력하여 `preprocess.py` 설계 전 데이터 구조를 파악한다.

## 실행

```bash
# 프로젝트 루트에서 실행
python src/check_data_structure.py
```

## 출력 항목

| 항목 | 설명 |
|------|------|
| 디렉토리 트리 | 최대 4단계까지 폴더/파일 구조 출력 |
| 확장자별 파일 수 | jpg/png/json/xml 등 파일 유형 파악 |
| JSON 라벨 샘플 | 첫 번째 JSON 파일 내용 일부 출력 |

## 전제 조건

- `dataset/raw/aihub_aging/` 디렉토리에 원본 데이터가 압축 해제된 상태여야 한다.
- 디렉토리가 없으면 에러 메시지를 출력하고 종료한다.
