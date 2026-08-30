import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path

other_dir = Path("datasets/processed/other")
other_dir.mkdir(parents=True, exist_ok=True)

print("Building non-aquatic dataset (chairs, objects, backgrounds)...")
count = 0

# 1. Fetch real-world non-aquatic images (objects, interiors, landscapes)
for i in range(1, 151):
    try:
        url = f"https://picsum.photos/224/224?random={i}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(str(other_dir / f"object_{i}.jpg"), img)
                count += 1
    except Exception:
        pass

# 2. Generate non-aquatic geometric/texture patterns (furniture wood grains, solid walls, random noise)
for i in range(1, 151):
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    bg_color = np.random.randint(40, 220, size=3).tolist()
    canvas[:] = bg_color
    
    # Render geometric lines, shapes, and noise resembling room corners/furniture
    for _ in range(np.random.randint(4, 12)):
        pt1 = tuple(np.random.randint(0, 224, size=2))
        pt2 = tuple(np.random.randint(0, 224, size=2))
        color = np.random.randint(0, 255, size=3).tolist()
        thickness = np.random.randint(1, 12)
        shape_type = np.random.choice(["rect", "circle", "line"])
        if shape_type == "rect":
            cv2.rectangle(canvas, pt1, pt2, color, -1 if np.random.rand() > 0.5 else thickness)
        elif shape_type == "circle":
            cv2.circle(canvas, pt1, np.random.randint(10, 80), color, -1)
        else:
            cv2.line(canvas, pt1, pt2, color, thickness)
            
    cv2.imwrite(str(other_dir / f"pattern_{i}.jpg"), canvas)
    count += 1

print(f"Successfully created {count} non-aquatic samples in datasets/processed/other")
