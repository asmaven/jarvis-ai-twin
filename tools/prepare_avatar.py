"""Turn Memoji screenshots into aligned, transparent avatar frames.

Usage: ./venv/bin/python tools/prepare_avatar.py
Inputs  (avatar/): IMG_6200 = neutral, IMG_6201 = talking (mouth open), IMG_6202 = big smile, IMG_6203 = blink
        IMG_6208 = "ahh" (open), IMG_6206 = "ooo" (rounded), IMG_6207 = lips pressed (m/b/p)
Outputs (static/avatar/): neutral.png, talk.png, smile.png, blink.png, and mouth frames talk/ee/aa/oo/press_on_neutral.png
         + avatar/preview.png (contact sheet)

Background removal only clears near-white pixels CONNECTED to the image border, so the white of the
eyes and teeth inside the face stays opaque. Frames are aligned on head width (ears are stable across
expressions) and the top of the hair, so only the mouth/eyes change between frames.
"""
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "avatar"
OUT = ROOT / "static" / "avatar"
FRAMES = {"neutral": "IMG_6200.jpeg", "talk": "IMG_6201.jpeg", "smile": "IMG_6202.jpeg", "blink": "IMG_6203.jpeg"}

CANVAS_W, CANVAS_H = 512, 560
HEAD_W = 400          # head width (ear to ear) in the output frame
TOP_MARGIN = 36       # gap above the hair


def background_mask(rgb: np.ndarray) -> np.ndarray:
    """True where the pixel is background: light, unsaturated, and connected to the border."""
    mx, mn = rgb.max(axis=2).astype(int), rgb.min(axis=2).astype(int)
    light = (mn > 215) & (mx - mn < 25)
    h, w = light.shape
    bg = np.zeros_like(light)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if light[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if light[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and light[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                q.append((ny, nx))
    return bg


def process(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    bg = background_mask(rgb)
    alpha = Image.fromarray(np.where(bg, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    rgba = img.copy()
    rgba.putalpha(alpha)

    ys, xs = np.nonzero(~bg)
    x0, x1, y0 = xs.min(), xs.max(), ys.min()
    scale = HEAD_W / (x1 - x0 + 1)
    resized = rgba.resize((round(rgba.width * scale), round(rgba.height * scale)), Image.LANCZOS)
    cx = (x0 + x1) / 2 * scale
    top = y0 * scale
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    canvas.alpha_composite(resized, (round(CANVAS_W / 2 - cx), round(TOP_MARGIN - top)))
    return canvas


NEUTRAL_MOUTH = (160, 390, 330, 420)   # resting lips incl. corners, in output-frame pixels

# Mouth frames change ONLY the lips area of the neutral face, so the jaw, cheeks, and eyes never move between
# frames (moving jawlines read as flicker/blurry lines when the mouth changes fast).
# name: (shot, mouth box in that shot (x0, y0, x1, y1), shift onto neutral's mouth, resting-lip area to cover)
MOUTH_SHOTS = {
    "talk":  ("IMG_6201.jpeg", (180, 402, 298, 442), (6, -4), NEUTRAL_MOUTH),     # half open
    "ee":    ("IMG_6202.jpeg", (172, 388, 322, 442), (0, 0), NEUTRAL_MOUTH),      # stretched (ee, s, f)
    "aa":    ("IMG_6208.jpeg", (202, 402, 315, 464), (-13, -2), (180, 396, 310, 416)),  # open "ah"
    "oo":    ("IMG_6206.jpeg", (218, 430, 288, 467), (-8, -26), NEUTRAL_MOUTH),   # rounded
    "press": ("IMG_6207.jpeg", (192, 404, 302, 418), (-2, -8), NEUTRAL_MOUTH),    # lips pressed (m, b, p)
}


def _ellipse(size, box, pad: int, feather: float) -> np.ndarray:
    m = Image.new("L", size, 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(m).ellipse((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=255)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m).astype(np.float32) / 255


def with_mouth(base: Image.Image, donor: Image.Image, box, shift, cover, pad: int = 16, feather: int = 7) -> Image.Image:
    """Paste the donor's mouth onto the base face with a soft edge, skin tone matched to the base."""
    dx, dy = shift
    moved = Image.new("RGBA", donor.size, (0, 0, 0, 0))
    moved.alpha_composite(donor, (dx, dy))
    b, d = np.asarray(base).astype(np.float32), np.asarray(moved).astype(np.float32)
    box = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
    mask = np.maximum(_ellipse(base.size, box, pad, feather), _ellipse(base.size, cover, 14, feather + 2))
    # Skin-tone correction that varies smoothly across the face: measure (base - donor) on skin both faces share,
    # skipping the lips, then fill it in under the lips by normalized blurring. The mouth then takes on exactly
    # the neutral face's lighting, so its colour doesn't jump when frames switch.
    lips = (_ellipse(base.size, box, 4, 0) > 0.5) | (_ellipse(base.size, cover, 0, 0) > 0.5)
    skin = ((b[..., 3] > 250) & (d[..., 3] > 250) & ~lips & (_ellipse(base.size, box, pad + 40, 0) > 0.5)).astype(np.float32)
    diff = (b[..., :3] - d[..., :3]) * skin[..., None]
    weight = _blur(skin, 14)[..., None]
    correction = np.stack([_blur(diff[..., c], 14) for c in range(3)], axis=-1) / np.maximum(weight, 1e-3)
    k = (mask * d[..., 3] / 255)[..., None]
    rgb = b[..., :3] * (1 - k) + np.clip(d[..., :3] + correction, 0, 255) * k
    return Image.fromarray(np.dstack([rgb, b[..., 3]]).astype(np.uint8))


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float array (separable, edges padded)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.pad(a, r, mode="edge")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, file in FRAMES.items():
        frames[name] = process(SRC / file)
        frames[name].save(OUT / f"{name}.png", optimize=True)
        print(f"saved {name}.png")
    for name, (file, box, shift, cover) in MOUTH_SHOTS.items():
        frames[f"{name}_on_neutral"] = with_mouth(frames["neutral"], process(SRC / file), box, shift, cover)
        frames[f"{name}_on_neutral"].save(OUT / f"{name}_on_neutral.png", optimize=True)
        print(f"saved {name}_on_neutral.png")
    sheet = Image.new("RGBA", (CANVAS_W * len(frames), CANVAS_H), (225, 232, 245, 255))
    for i, frame in enumerate(frames.values()):
        sheet.alpha_composite(frame, (i * CANVAS_W, 0))
    sheet.convert("RGB").save(SRC / "preview.png")
    print("saved avatar/preview.png")


if __name__ == "__main__":
    main()
