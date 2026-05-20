"""
Mejora calidad v2: parte de las imágenes ya mejoradas en ./mejoradas/,
hace una segunda pasada por Gemini con prompt más agresivo y luego un
upscale 2x con Lanczos para subir resolución real. Resultado en ./mejoradas-v2/.
"""

import io
import sys
import time
import mimetypes
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).parent
API_KEY_FILE = BASE_DIR / "gemini.txt"
SRC_DIR = BASE_DIR / "mejoradas"
OUT_DIR = BASE_DIR / "mejoradas-v2"

MODEL = "gemini-2.5-flash-image"

PROMPT = (
    "Take this photograph and produce an ULTRA high-quality, magazine-grade, "
    "photorealistic master version. Maximize resolution and sharpness. "
    "Remove ALL JPEG compression artifacts, banding, blur and noise. "
    "Reconstruct realistic fine detail: skin pores and hair strands on people, "
    "wood grain, fabric weave, leaf veins, stone texture, brick mortar lines, "
    "individual blades of grass, water ripples — whatever applies to the scene. "
    "Recover crisp edges on architecture and clean text/logos if present. "
    "Improve dynamic range: deeper blacks, cleaner highlights, rich natural color, "
    "white-balanced, professional cinematic lighting. "
    "Keep the EXACT same composition, framing, subject, perspective, pose, "
    "clothing, objects, background and time of day. Do NOT add or remove anything, "
    "do NOT change identities, do NOT crop. Only enhance. "
    "Output a single photographic image at the highest resolution possible."
)

UPSCALE_FACTOR = 2  # Lanczos upscale tras Gemini


def read_api_key() -> str:
    return API_KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()


def enhance(client: genai.Client, data: bytes, mime: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=data, mime_type=mime), PROMPT],
    )
    for cand in response.candidates or []:
        if not cand.content or not cand.content.parts:
            continue
        for part in cand.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data, inline.mime_type or "image/png"
    return None


def lanczos_upscale(data: bytes, factor: int) -> bytes:
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    new = img.resize((w * factor, h * factor), Image.LANCZOS)
    buf = io.BytesIO()
    new.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def main() -> int:
    if not SRC_DIR.exists():
        print(f"No existe {SRC_DIR}. Ejecuta primero mejorar.py")
        return 1

    api_key = read_api_key()
    client = genai.Client(api_key=api_key)
    OUT_DIR.mkdir(exist_ok=True)

    files = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not files:
        print(f"No hay imágenes en {SRC_DIR}")
        return 1

    ok = 0
    for i, src in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {src.name}")
        try:
            data = src.read_bytes()
            mime = mimetypes.guess_type(src.name)[0] or "image/png"
            print(f"   entrada {len(data)//1024} KB ({mime})")

            result = enhance(client, data, mime)
            if not result:
                print("   Gemini no devolvió imagen, salto")
                continue
            gem_bytes, gem_mime = result
            print(f"   pasada Gemini {len(gem_bytes)//1024} KB ({gem_mime})")

            final_bytes = lanczos_upscale(gem_bytes, UPSCALE_FACTOR)
            out = OUT_DIR / (src.stem.replace("_hq", "") + f"_hq2x{UPSCALE_FACTOR}.png")
            out.write_bytes(final_bytes)

            with Image.open(io.BytesIO(final_bytes)) as im:
                w, h = im.size
            print(f"   guardada -> {out.name} ({len(final_bytes)//1024} KB, {w}x{h})")
            ok += 1
        except Exception as e:
            print(f"   error: {e}")
        time.sleep(1)

    print(f"\nListo v2: {ok}/{len(files)} en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
