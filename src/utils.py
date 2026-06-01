import numpy as np
from pathlib import Path

MAX_DIM = 1280


def extract_face_crop(img_path: str, max_dim: int = MAX_DIM) -> np.ndarray:
    """resize -> opencv detect -> 20% padding crop -> fallback: full image. Returns RGB np.ndarray."""
    from deepface import DeepFace
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    img_arr = np.array(img)
    sw, sh = img_arr.shape[1], img_arr.shape[0]

    try:
        faces = DeepFace.extract_faces(img_path=img_arr, detector_backend="opencv", enforce_detection=True)
        if faces:
            fa = faces[0]["facial_area"]
            x, y, fw, fh = fa["x"], fa["y"], fa["w"], fa["h"]
            px, py = fw * 0.2, fh * 0.2
            x1, y1 = max(0, int(x - px)), max(0, int(y - py))
            x2, y2 = min(sw, int(x + fw + px)), min(sh, int(y + fh + py))
            if x2 > x1 and y2 > y1:
                return img_arr[y1:y2, x1:x2]
    except Exception:
        pass
    return img_arr
