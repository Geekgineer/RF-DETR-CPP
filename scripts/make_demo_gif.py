"""
Generate data/demo.gif — 3-task looping demo (detection · segmentation · keypoint).
Uses the existing dog.jpg / dog_rfdetr.jpg assets.  No model inference needed.
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ── palette ──────────────────────────────────────────────────────────────────
DARK    = (13,  17,  23)
DARK2   = (22,  27,  34)
ORANGE  = (249, 115,  22)
ORANGE2 = (234,  88,  12)
BLUE    = ( 59, 130, 246)
BLUE2   = ( 30,  64, 175)
BLUE_L  = ( 96, 165, 250)
PURPLE  = (168,  85, 247)
PURPLE2 = ( 76,  29, 149)
PURPLE_L= (192, 132, 252)
WHITE   = (255, 255, 255)
GREY    = (139, 148, 158)
GREEN   = ( 34, 197,  94)

# ── layout ───────────────────────────────────────────────────────────────────
SCALE   = 1          # keep native 302×329 — GIF palette needs small size
TOP_H   = 36         # task-name bar height
BOT_H   = 28         # stats bar height
IMG_W   = 302 * SCALE
IMG_H   = 329 * SCALE
FULL_W  = IMG_W
FULL_H  = IMG_H + TOP_H + BOT_H

# bounding box from dog_rfdetr.json (scaled)
BX = 0   * SCALE
BY = 53  * SCALE
BW = 265 * SCALE
BH = 275 * SCALE

# ── helpers ───────────────────────────────────────────────────────────────────
def load(name: str) -> Image.Image:
    return Image.open(DATA / name).convert("RGB").resize(
        (IMG_W, IMG_H), Image.LANCZOS)


def pill(draw: ImageDraw.ImageDraw, xy, text, bg, fg=WHITE, r=14, pad=8):
    x, y = xy
    w = draw.textlength(text) + pad * 2
    h = r * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=bg)
    draw.text((x + pad, y + (h - 12) // 2), text, fill=fg)


def corner_mark(draw, x1, y1, x2, y2, color, t=3, sz=28):
    """Draw L-shaped corner markers on a rectangle."""
    for ax, ay, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        draw.line([ax, ay, ax + dx*sz, ay],              fill=color, width=t)
        draw.line([ax, ay, ax,          ay + dy*sz],     fill=color, width=t)


def dashed_rect(draw, x1, y1, x2, y2, color, t=2, dash=10, gap=6):
    for axis in ("h", "v"):
        if axis == "h":
            for y in (y1, y2):
                x = x1
                while x < x2:
                    draw.line([x, y, min(x+dash, x2), y], fill=color, width=t)
                    x += dash + gap
        else:
            for x in (x1, x2):
                y = y1
                while y < y2:
                    draw.line([x, y, x, min(y+dash, y2)], fill=color, width=t)
                    y += dash + gap


def make_base(img_rgb: Image.Image, task: str, stats: str, accent) -> Image.Image:
    frame = Image.new("RGB", (FULL_W, FULL_H), DARK)
    # top bar
    top = Image.new("RGB", (FULL_W, TOP_H), DARK2)
    d = ImageDraw.Draw(top)
    # accent stripe on left
    d.rectangle([0, 0, 4, TOP_H], fill=accent)
    d.text((16, 14), f"RF-DETR-CPP", fill=WHITE)
    d.text((110, 14), f"· {task}", fill=accent)
    frame.paste(top, (0, 0))
    # image
    frame.paste(img_rgb, (0, TOP_H))
    # bottom bar
    bot = Image.new("RGB", (FULL_W, BOT_H), DARK2)
    d2 = ImageDraw.Draw(bot)
    d2.text((12, 12), stats, fill=GREY)
    frame.paste(bot, (0, TOP_H + IMG_H))
    return frame


# ── task 1: Detection ─────────────────────────────────────────────────────────
def det_frame() -> Image.Image:
    img = load("dog_rfdetr.jpg")   # already has box drawn
    base = make_base(img, "Detection",
                     "dog 0.977  ·  rfdetr-base  ·  CPU  ·  399 ms avg", ORANGE)
    d = ImageDraw.Draw(base)
    # pill on top of the image area
    pill(d, (BX + 2, TOP_H + BY - 22), "dog  0.977", ORANGE)
    return base


# ── task 2: Segmentation ─────────────────────────────────────────────────────
def seg_frame() -> Image.Image:
    img = load("dog.jpg")
    # blue semi-transparent mask layer
    overlay = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([BX, BY, BX+BW, BY+BH], fill=(*BLUE, 55))

    # composite
    base_rgba = img.convert("RGBA")
    composite = Image.alpha_composite(base_rgba, overlay).convert("RGB")

    # dashed outline + corner marks
    cd = ImageDraw.Draw(composite)
    dashed_rect(cd, BX, BY, BX+BW, BY+BH, (*BLUE_L, 200))
    corner_mark(cd, BX, BY, BX+BW, BY+BH, BLUE, t=3, sz=24)
    # label
    pill(cd, (BX + 2, BY - 24), "mask  dog", BLUE2, BLUE_L)

    base = make_base(composite, "Segmentation",
                     "rfdetr-seg-nano  ·  mask H/4×W/4  ·  CPU", BLUE)
    return base


# ── task 3: Keypoint ─────────────────────────────────────────────────────────
def kp_frame() -> Image.Image:
    img = load("dog.jpg")

    # Approximate dog keypoints at 2× scale (tuned to the 302×329 image)
    s = SCALE
    kps = {
        "nose":       (110*s,  82*s),
        "l_eye":      ( 92*s,  72*s),
        "r_eye":      (128*s,  70*s),
        "l_ear":      ( 72*s,  62*s),
        "r_ear":      (148*s,  60*s),
        "neck":       (118*s, 120*s),
        "l_shoulder": ( 68*s, 155*s),
        "r_shoulder": (175*s, 155*s),
        "l_elbow":    ( 52*s, 215*s),
        "r_elbow":    (185*s, 220*s),
        "l_wrist":    ( 44*s, 270*s),
        "r_wrist":    (190*s, 278*s),
        "l_hip":      ( 82*s, 238*s),
        "r_hip":      (195*s, 238*s),
        "l_knee":     ( 68*s, 286*s),
        "r_knee":     (210*s, 286*s),
        "l_ankle":    ( 58*s, 320*s),
        "r_ankle":    (220*s, 322*s),
    }

    SKEL = [
        ("nose",   "l_eye"),  ("nose",   "r_eye"),
        ("l_eye",  "l_ear"),  ("r_eye",  "r_ear"),
        ("nose",   "neck"),
        ("neck",   "l_shoulder"), ("neck",   "r_shoulder"),
        ("l_shoulder","l_elbow"), ("l_elbow","l_wrist"),
        ("r_shoulder","r_elbow"), ("r_elbow","r_wrist"),
        ("l_shoulder","l_hip"),   ("r_shoulder","r_hip"),
        ("l_hip",  "r_hip"),
        ("l_hip",  "l_knee"),  ("l_knee",  "l_ankle"),
        ("r_hip",  "r_knee"),  ("r_knee",  "r_ankle"),
    ]

    # draw on RGBA so we can use alpha on limb lines
    rgba = img.convert("RGBA")
    limb_layer = Image.new("RGBA", (IMG_W, IMG_H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(limb_layer)
    for a, b in SKEL:
        if a in kps and b in kps:
            ld.line([kps[a], kps[b]], fill=(*PURPLE, 160), width=3)
    composite = Image.alpha_composite(rgba, limb_layer).convert("RGB")

    kd = ImageDraw.Draw(composite)
    for name, pt in kps.items():
        r = 5
        kd.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=PURPLE_L, outline=PURPLE2, width=1)

    # bounding box corners
    corner_mark(kd, BX, BY, BX+BW, BY+BH, PURPLE, t=3, sz=24)
    pill(kd, (BX + 2, BY - 24), "person  0.91  17 kp", PURPLE2, PURPLE_L)

    base = make_base(composite, "Keypoint",
                     "rfdetr-keypoint-preview  ·  17 COCO kp/person  ·  CPU", PURPLE)
    return base


# ── build GIF ────────────────────────────────────────────────────────────────
def crossfade(a: Image.Image, b: Image.Image, steps=6):
    arr_a = [f for f in [a] * 1]   # unused — just yield blend frames
    imgs = []
    for i in range(1, steps):
        alpha = i / steps
        imgs.append(Image.blend(a, b, alpha))
    return imgs


def main():
    print("Building frames …")
    d = det_frame()
    s = seg_frame()
    k = kp_frame()

    # Each task: 1 frame @ 2600 ms still; 4-frame crossfade @ 60 ms
    STILL_N    = 1
    STILL_MS   = 2600
    FADE_N     = 4
    FADE_MS    = 60

    frames, durations = [], []

    for fr_from, fr_to in [(d, s), (s, k), (k, d)]:
        for _ in range(STILL_N):
            frames.append(fr_from)
            durations.append(STILL_MS)
        for blend in crossfade(fr_from, fr_to, FADE_N + 1):
            frames.append(blend)
            durations.append(FADE_MS)

    out = DATA / "demo.gif"
    print(f"Saving {len(frames)} frames → {out}")
    # Quantize to 128 colours for smaller GIF — convert back to P-mode per frame
    pal_frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=0)
                  for f in frames]
    pal_frames[0].save(
        out,
        save_all=True,
        append_images=pal_frames[1:],
        optimize=True,
        duration=durations,
        loop=0,
    )
    size_kb = out.stat().st_size // 1024
    print(f"Done — {FULL_W}×{FULL_H} px, {size_kb} KB, {len(frames)} frames")


if __name__ == "__main__":
    main()
