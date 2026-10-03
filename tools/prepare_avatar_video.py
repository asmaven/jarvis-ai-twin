"""Build the avatar frames from one Memoji video (consistent lighting and pose in every frame).

Usage: ./venv/bin/python tools/prepare_avatar_video.py
Input:  avatar/EmojiMovie812434092.mov (Memoji recording on a black background, 60 fps)
Output: static/avatar_video/ with the same frame names as static/avatar/ (neutral, blink, smile, and the
        mouth frames talk/ee/aa/oo/press_on_neutral.png) + avatar/preview_video.png

Every frame is scaled to the same head width and aligned on the nose, then only the mouth area of the
chosen frame is blended onto the neutral frame, so eyes, jaw, and skin tone never change between mouths.
Needs: pip install imageio-ffmpeg (bundles ffmpeg).
"""
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "avatar" / "EmojiMovie812434092.mov"
OUT = ROOT / "static" / "avatar_video"
FPS = 30

CANVAS_W, CANVAS_H = 512, 560
HEAD_W = 400
TOP_MARGIN = 36

# Frame numbers (at 30 fps) picked by eye from the recording.
NEUTRAL, BLINK, SMILE = 100, 2, 56
MOUTHS = {            # name: frame
    "talk": 166,      # slightly open (most consonants, ee, s)
    "aa": 234,        # open "ah"
    "oo": 246,        # rounded "oo" / "oh"
    "press": 146,     # lips together (m, b, p, f, v)
    "ee": 298,        # teeth showing (not used in the lip-sync map today)
}
MOUTH_AREA = (170, 370, 330, 450)   # x0, y0, x1, y1 on the normalized canvas
NOSE_AREA = (195, 300, 300, 375)    # used to line frames up


def extract_frames(tmp: Path) -> list[Path]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(VIDEO), "-vf", f"fps={FPS}", str(tmp / "f_%04d.png")],
                   check=True)
    return sorted(tmp.glob("f_*.png"))


def background(rgb: np.ndarray) -> np.ndarray:
    """True where the pixel is the black background (near-black and connected to the border)."""
    dark = rgb.max(axis=2) < 6
    h, w = dark.shape
    bg = np.zeros_like(dark)
    q = deque((y, x) for y in range(h) for x in (0, w - 1) if dark[y, x])
    q.extend((y, x) for x in range(w) for y in (0, h - 1) if dark[y, x])
    for y, x in q:
        bg[y, x] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and dark[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                q.append((ny, nx))
    return bg


def normalize(path: Path) -> Image.Image:
    """Cut the head out of the black background and place it at a fixed size and position."""
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img).astype(np.float32)
    bg = background(rgb.astype(np.uint8))
    # Soft edge: the video is composited on black, so edge pixels are colour * coverage.
    near_bg = np.asarray(Image.fromarray((bg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0
    alpha = np.where(bg, 0.0, 1.0)
    edge = near_bg & ~bg
    alpha[edge] = np.clip(rgb[edge].max(axis=1) / 45.0, 0, 1)
    rgb[edge] = rgb[edge] / np.maximum(alpha[edge][:, None], 1e-3)
    rgba = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8))

    ys, xs = np.nonzero(~bg)
    x0, x1, y0 = xs.min(), xs.max(), ys.min()
    scale = HEAD_W / (x1 - x0 + 1)
    resized = rgba.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    resized.putalpha(resized.getchannel("A").filter(ImageFilter.GaussianBlur(1.2)))  # smooth the upscaled outline
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    canvas.alpha_composite(resized, (round(CANVAS_W / 2 - (x0 + x1) / 2 * scale), round(TOP_MARGIN - y0 * scale)))
    return canvas


def align(base: Image.Image, frame: Image.Image, r: int = 14) -> Image.Image:
    """Shift the frame so its nose lines up with the base frame's nose."""
    b = np.asarray(base.convert("L")).astype(np.float32)
    f = np.asarray(frame.convert("L")).astype(np.float32)
    x0, y0, x1, y1 = NOSE_AREA
    ref = b[y0:y1, x0:x1]
    _, dx, dy = min((float(np.mean((f[y0 + dy:y1 + dy, x0 + dx:x1 + dx] - ref) ** 2)), dx, dy)
                    for dy in range(-r, r + 1) for dx in range(-r, r + 1))
    out = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    out.alpha_composite(frame, (-dx, -dy))
    return out


def with_mouth(base: Image.Image, donor: Image.Image, feather: int = 8) -> Image.Image:
    """Blend the donor's mouth area onto the base face (same video, so colours already match)."""
    mask = Image.new("L", base.size, 0)
    ImageDraw.Draw(mask).ellipse(MOUTH_AREA, fill=255)
    out = base.copy()
    out.paste(donor, (0, 0), mask.filter(ImageFilter.GaussianBlur(feather)))
    out.putalpha(base.getchannel("A"))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = extract_frames(Path(tmp))
        frame = lambda n: normalize(paths[n])  # noqa: E731
        neutral = frame(NEUTRAL)
        frames = {"neutral": neutral, "blink": align(neutral, frame(BLINK)), "smile": align(neutral, frame(SMILE))}
        for name, n in MOUTHS.items():
            frames[f"{name}_on_neutral"] = with_mouth(neutral, align(neutral, frame(n)))
    for name, img in frames.items():
        img.save(OUT / f"{name}.png", optimize=True)
        print(f"saved {name}.png")
    sheet = Image.new("RGBA", (CANVAS_W * len(frames), CANVAS_H), (225, 232, 245, 255))
    for i, img in enumerate(frames.values()):
        sheet.alpha_composite(img, (i * CANVAS_W, 0))
    sheet.convert("RGB").save(ROOT / "avatar" / "preview_video.png")
    print("saved avatar/preview_video.png")


if __name__ == "__main__":
    main()
