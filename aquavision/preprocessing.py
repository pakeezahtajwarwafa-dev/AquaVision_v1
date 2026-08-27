import cv2
import numpy as np

def pad_to_square(img: np.ndarray, fill_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Pads an image with a solid color to make it square while keeping content centered."""
    h, w = img.shape[:2]
    if h == w:
        return img

    max_side = max(h, w)
    top = (max_side - h) // 2
    bottom = max_side - h - top
    left = (max_side - w) // 2
    right = max_side - w - left

    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=fill_color
    )

def letterbox_resize(img: np.ndarray, target_size: int = 224, species: str = "fish") -> np.ndarray:
    """Pads image to square and resizes using optimal interpolation based on up/downscaling."""
    # Step 1: Select species-aware background fill
    fill_color = (255, 255, 255) if species.lower() == "shrimp" else (0, 0, 0)
    
    # Step 2: Pad canvas to square (1:1 aspect ratio)
    squared_img = pad_to_square(img, fill_color=fill_color)
    
    # Step 3: Choose interpolation method
    orig_side = squared_img.shape[0]
    if orig_side > target_size:
        interp = cv2.INTER_AREA    # Downscaling (avoids aliasing)
    else:
        interp = cv2.INTER_CUBIC   # Upscaling (smoother textures)

    return cv2.resize(squared_img, (target_size, target_size), interpolation=interp)
