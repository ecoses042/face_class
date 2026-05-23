"""
데이터 다운로드 후 실행: raw 데이터 구조 파악 및 요약 출력
"""
import os
import json
import sys
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("dataset/raw/aihub_aging")


def count_files_by_ext(root: Path) -> dict:
    counts = defaultdict(int)
    for p in root.rglob("*"):
        if p.is_file():
            counts[p.suffix.lower()] += 1
    return dict(counts)


def sample_json(root: Path, n: int = 1) -> list[dict]:
    samples = []
    for p in root.rglob("*.json"):
        try:
            with open(p) as f:
                samples.append({"file": str(p), "content": json.load(f)})
        except Exception:
            continue
        if len(samples) >= n:
            break
    return samples


def print_tree(root: Path, prefix: str = "", max_depth: int = 4, depth: int = 0):
    if depth >= max_depth:
        return
    entries = sorted(root.iterdir()) if root.is_dir() else []
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    shown = dirs + files[:5]
    hidden = len(files) - 5 if len(files) > 5 else 0
    for i, entry in enumerate(shown):
        connector = "└── " if i == len(shown) - 1 and not hidden else "├── "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(shown) - 1 and not hidden else "│   "
            print_tree(entry, prefix + extension, max_depth, depth + 1)
    if hidden:
        print(prefix + f"└── ... ({hidden} more files)")


def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] {RAW_DIR} 디렉토리가 없습니다. 데이터를 먼저 다운로드하세요.")
        sys.exit(1)

    print("=" * 60)
    print(f"디렉토리 트리 (최대 4단계): {RAW_DIR}")
    print("=" * 60)
    print_tree(RAW_DIR)

    print("\n" + "=" * 60)
    print("확장자별 파일 수")
    print("=" * 60)
    for ext, cnt in sorted(count_files_by_ext(RAW_DIR).items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no ext)':15s}: {cnt:,}")

    print("\n" + "=" * 60)
    print("JSON 라벨 샘플 (1개)")
    print("=" * 60)
    samples = sample_json(RAW_DIR, n=1)
    if samples:
        s = samples[0]
        print(f"파일: {s['file']}")
        print(json.dumps(s["content"], ensure_ascii=False, indent=2)[:2000])
    else:
        print("JSON 파일을 찾지 못했습니다.")

    print("\n[완료] 위 구조를 확인하고 src/preprocess.py 설계에 활용하세요.")


if __name__ == "__main__":
    main()
