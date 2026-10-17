"""Black-and-white drawing augmentations for the parser (CPU, numpy + OpenCV).

The parser must read real scanned or exported drawings: any line width, blur,
noise, JPEG artefacts, contrast, pure black-and-white. Geometric changes (scale,
90-degree rotation, flip) are applied to image AND label; photometric ones only
to the image. Images are greyscale uint8 (0 = black ink, 255 = white paper).
"""

import cv2
import numpy as np


def rot_flip(img, lbl, rng, rot90=True, flip=True):
    k = int(rng.integers(4)) if rot90 else 0
    img, lbl = np.rot90(img, k), np.rot90(lbl, k)
    if flip and rng.random() < 0.5:
        img, lbl = img[:, ::-1], lbl[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)


def binarise(img, rng, offset):
    """Pure black and white: Otsu threshold shifted by a random offset."""
    t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = np.clip(t + rng.uniform(*offset), 1, 254)
    return np.where(img > t, 255, 0).astype(np.uint8)


def line_width(img, rng, px, keep_ink=0.6):
    """Thicken (min filter grows dark ink) or thin (max filter shrinks it) lines by 1-2 px.

    Thinning is undone if it would erase more than 40% of the ink: drawings made of
    1-px outlines would otherwise vanish while their labels still show walls.
    """
    n = int(rng.integers(px[0], px[1] + 1))
    kernel = np.ones((n + 1, n + 1), np.uint8)       # changes every line's width by n px
    if rng.random() < 0.5:
        return cv2.erode(img, kernel)
    thin = cv2.dilate(img, kernel)
    ink_before, ink_after = (img < 128).sum(), (thin < 128).sum()
    return thin if ink_after >= keep_ink * ink_before else img


def blur(img, rng, sigma):
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(*sigma))


def noise(img, rng, std):
    n = rng.normal(0, rng.uniform(*std), img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def jpeg(img, rng, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(quality[0], quality[1] + 1))])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else img


def contrast(img, rng, alpha, beta):
    return np.clip(img.astype(np.float32) * rng.uniform(*alpha) + rng.uniform(*beta), 0, 255).astype(np.uint8)


def photometric(img, rng, cfg):
    """Apply each photometric augmentation with its own probability, in a fixed order."""
    if rng.random() < cfg["line_width"]["p"]:
        img = line_width(img, rng, cfg["line_width"]["px"])
    if rng.random() < cfg["blur"]["p"]:
        img = blur(img, rng, cfg["blur"]["sigma"])
    if rng.random() < cfg["contrast"]["p"]:
        img = contrast(img, rng, cfg["contrast"]["alpha"], cfg["contrast"]["beta"])
    if rng.random() < cfg["binarise"]["p"]:
        img = binarise(img, rng, cfg["binarise"]["offset"])
    if rng.random() < cfg["noise"]["p"]:
        img = noise(img, rng, cfg["noise"]["std"])
    if rng.random() < cfg["jpeg"]["p"]:
        img = jpeg(img, rng, cfg["jpeg"]["quality"])
    return img
