"""
Mejora v3 — look de fotografía profesional.
Pipeline por imagen:
  1) Pasada A (Gemini): limpieza estructural — quita artefactos, reconstruye detalle.
  2) Pasada B (Gemini): grading profesional — luz, color, contraste, textura real.
  3) Upscale 2x con Lanczos.
  4) Sharpening sutil (unsharp mask).
Lee originales de ./originales/ y guarda en ./mejoradas-v3/.
"""

import io
import sys
import time
import mimetypes
from pathlib import Path

from PIL import Image, ImageFilter
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).parent
API_KEY_FILE = BASE_DIR / "gemini.txt"
SRC_DIR = BASE_DIR / "originales"
OUT_DIR = BASE_DIR / "mejoradas-v3"

MODEL = "gemini-2.5-flash-image"

PROMPT_A = (
    "Restore this photograph as a professional photo retoucher would. "
    "Completely remove JPEG compression artifacts, blockiness, halos, banding, "
    "color noise and chroma smearing. Recover lost micro-detail in EVERY surface: "
    "skin pores, hair strands, fabric weave, wood grain, stone, brick, foliage, "
    "asphalt, water. Rebuild sharp clean edges on architecture and on the horizon. "
    "Maintain the exact same composition, framing, subject, perspective and content. "
    "Do not add or remove elements, do not change identities, do not crop, do not stylize. "
    "Output only a clean, neutral, high-fidelity photographic image."
)

PROMPT_B = (
    "Apply a professional photographer's color grade and lighting pass to this image, "
    "as if shot with a full-frame DSLR (Canon R5 / Sony A7R) on a prime lens, "
    "RAW developed in Lightroom by a pro. "
    "Cinematic dynamic range: deep but detailed shadows, soft natural highlights, "
    "no clipping, no blown skies. "
    "Rich, accurate, white-balanced colors with subtle film-like tonality. "
    "Crystal-clear focus on the main subject, natural micro-contrast, "
    "realistic ambient occlusion, lifelike global illumination. "
    "Make it look like a real professional photograph fit for a magazine spread or "
    "a luxury real-estate / travel brochure. "
    "Keep the EXACT same composition, framing, subject, perspective, time of day, "
    "objects and background. Do NOT add, remove, stylize or hallucinate elements. "
    "Output one photorealistic image only, highest possible resolution."
)

UPSCALE_FACTOR = 2


def read_api_key() -> str:
    return API_KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()


def gemini_pass(client: genai.Client, data: bytes, mime: str, prompt: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
    )
    for cand in response.candidates or []:
        if not cand.content or not cand.content.parts:
            continue
        for part in cand.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data, inline.mime_type or "image/png"
    return None


def lanczos_upscale_and_sharpen(data: bytes, factor: int) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    big = img.resize((w * factor, h * factor), Image.LANCZOS)
    sharpened = big.filter(ImageFilter.UnsharpMask(radius=1.4, percent=110, threshold=2))
    buf = io.BytesIO()
    sharpened.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def process(client: genai.Client, src: Path, out_dir: Path) -> bool:
    data = src.read_bytes()
    mime = mimetypes.guess_type(src.name)[0] or "image/jpeg"
    print(f"   entrada {len(data)//1024} KB ({mime})")

    res_a = gemini_pass(client, data, mime, PROMPT_A)
    if not res_a:
        print("   pasada A: sin imagen")
        return False
    a_bytes, a_mime = res_a
    print(f"   pasada A OK {len(a_bytes)//1024} KB")

    res_b = gemini_pass(client, a_bytes, a_mime, PROMPT_B)
    if not res_b:
        print("   pasada B falló, uso pasada A")
        b_bytes, b_mime = a_bytes, a_mime
    else:
        b_bytes, b_mime = res_b
        print(f"   pasada B OK {len(b_bytes)//1024} KB")

    final_bytes = lanczos_upscale_and_sharpen(b_bytes, UPSCALE_FACTOR)
    with Image.open(io.BytesIO(final_bytes)) as im:
        w, h = im.size

    stem = src.stem
    out = out_dir / f"{stem}_pro{w}x{h}.png"
    out.write_bytes(final_bytes)
    print(f"   guardada -> {out.name} ({len(final_bytes)//1024} KB, {w}x{h})")
    return True


def main() -> int:
    if not SRC_DIR.exists():
        print(f"No existe {SRC_DIR}. Ejecuta primero mejorar.py para descargar originales.")
        return 1

    api_key = read_api_key()
    client = genai.Client(api_key=api_key)
    OUT_DIR.mkdir(exist_ok=True)

    files = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        print(f"No hay imágenes en {SRC_DIR}")
        return 1

    ok = 0
    for i, src in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {src.name}")
        try:
            if process(client, src, OUT_DIR):
                ok += 1
        except Exception as e:
            print(f"   error: {e}")
        time.sleep(1)

    print(f"\nListo v3: {ok}/{len(files)} en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
