"""
Mejora la calidad de imágenes usando Gemini 2.5 Flash Image (Nano Banana).
Lee URLs desde urls.txt, descarga cada imagen, le pide a Gemini que la mejore
y guarda el resultado en la carpeta ./mejoradas/
"""

import os
import sys
import time
import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from google import genai
from google.genai import types

BASE_DIR = Path(__file__).parent
API_KEY_FILE = BASE_DIR / "gemini.txt"
URLS_FILE = BASE_DIR / "urls.txt"
OUT_DIR = BASE_DIR / "mejoradas"
ORIG_DIR = BASE_DIR / "originales"

MODEL = "gemini-2.5-flash-image"

PROMPT = (
    "Enhance this photograph to professional, high-resolution quality. "
    "Increase sharpness and clarity, remove compression artifacts and noise, "
    "improve lighting and color balance, recover fine detail in textures, "
    "and produce a clean, crisp, photorealistic result. "
    "Keep the exact same composition, subject, framing and content — "
    "do not add, remove or alter any element. Output only the improved image."
)


def read_api_key() -> str:
    key = API_KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not key:
        raise SystemExit("gemini.txt está vacío")
    return key


def read_urls() -> list[str]:
    lines = URLS_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def download_image(url: str) -> tuple[bytes, str]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
        mime = resp.headers.get_content_type() or "image/jpeg"
    return data, mime


def save_originally(idx: int, url: str, data: bytes) -> Path:
    ORIG_DIR.mkdir(exist_ok=True)
    name = Path(urlparse(url).path).name or f"img_{idx}.jpg"
    path = ORIG_DIR / f"{idx:02d}_{name}"
    path.write_bytes(data)
    return path


def enhance_image(client: genai.Client, data: bytes, mime: str) -> tuple[bytes, str] | None:
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            PROMPT,
        ],
    )
    for cand in response.candidates or []:
        if not cand.content or not cand.content.parts:
            continue
        for part in cand.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                out_mime = inline.mime_type or "image/png"
                return inline.data, out_mime
    return None


def ext_from_mime(mime: str) -> str:
    ext = mimetypes.guess_extension(mime or "")
    if not ext or ext == ".jpe":
        ext = ".png"
    return ext


def main() -> int:
    api_key = read_api_key()
    urls = read_urls()
    if not urls:
        print("No hay URLs en urls.txt")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    client = genai.Client(api_key=api_key)

    ok = 0
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            data, mime = download_image(url)
            print(f"   descargada ({len(data)//1024} KB, {mime})")
            save_originally(i, url, data)

            result = enhance_image(client, data, mime)
            if not result:
                print("   sin imagen devuelta, salto")
                continue

            out_bytes, out_mime = result
            stem = Path(urlparse(url).path).stem or f"img_{i}"
            out_path = OUT_DIR / f"{i:02d}_{stem}_hq{ext_from_mime(out_mime)}"
            out_path.write_bytes(out_bytes)
            print(f"   guardada -> {out_path.name} ({len(out_bytes)//1024} KB)")
            ok += 1
        except Exception as e:
            print(f"   error: {e}")
        time.sleep(1)

    print(f"\nListo: {ok}/{len(urls)} imágenes mejoradas en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
