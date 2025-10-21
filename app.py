import io, uuid, os
from typing import List
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANVAS_W, CANVAS_H = 200, 200
PADDING = 8
UPSCALE = 4
OPENAI_MODEL = "gpt-4o-mini"
PAGE_STORE: dict[str, list[bytes]] = {}

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    USE_RESPONSES = True
except Exception:
    USE_RESPONSES = False

PROMPT_ES = (
    "Lee la imagen y extrae el problema de cálculo/EDO. "
    "Resuélvelo paso a paso y devuelve SOLO un bloque LaTeX con display math, "
    "usando el entorno aligned. No agregues nada fuera del bloque. "
    "Reglas para pantalla 200x200: pasos cortos; divide con \\\\ si hace falta.\n\n"
    "Plantilla EXACTA:\n"
    "\\[\n"
    "\\begin{aligned}\n"
    "% líneas con \\\\ por salto\n"
    "\\end{aligned}\n"
    "\\]\n"
)

def ask_openai_vision(image_bytes: bytes) -> str:
    if not USE_RESPONSES:
        raise HTTPException(500, "SDK de OpenAI no disponible en entorno")
    msg = [
        {"role": "user", "content": [
            {"type":"input_text","text":PROMPT_ES},
            {"type":"input_image","image_bytes": image_bytes}
        ]}
    ]
    res = client.responses.create(model=OPENAI_MODEL, input=msg)
    return res.output_text.strip()

def latex_block_to_rgba(latex_block: str, scale=UPSCALE) -> Image.Image:
    clean = latex_block.replace("\\[", "").replace("\\]", "").strip()
    fig = plt.figure(figsize=(3, 12), dpi=100*scale)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.02, 0.98, f"$\\displaystyle {clean}$",
            ha="left", va="top", fontsize=24*scale)
    buf = io.BytesIO(); plt.savefig(buf, format="png", transparent=True, dpi=100*scale)
    plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img

def compose_to_pages(rgba: Image.Image) -> List[Image.Image]:
    white_bg = Image.new("RGBA", rgba.size, (255,255,255,255))
    imgL = Image.alpha_composite(white_bg, rgba).convert("L")
    max_w = CANVAS_W - 2*PADDING
    scale = min(1.0, max_w / imgL.width)
    new_w, new_h = int(imgL.width*scale), int(imgL.height*scale)
    imgL = imgL.resize((new_w, new_h), Image.LANCZOS)
    page_h = CANVAS_H - 2*PADDING

    pages = []
    y = 0
    while y < new_h:
        slice_h = min(page_h, new_h - y)
        crop = imgL.crop((0, y, new_w, y + slice_h))
        canvas = Image.new("L", (CANVAS_W, CANVAS_H), 255)
        canvas.paste(crop, ((CANVAS_W - new_w)//2, PADDING))
        canvas = ImageOps.unsharp_mask(canvas, radius=1, percent=130, threshold=2)
        pages.append(canvas); y += slice_h
    return pages

def to_pbm_p4(imgL: Image.Image) -> bytes:
    mono = imgL.convert("1", dither=Image.FLOYDSTEINBERG)
    w, h = mono.size
    return f"P4\n{w} {h}\n".encode("ascii") + mono.tobytes()

app = FastAPI()

class SolveResponse(BaseModel):
    token: str
    pages: int

@app.post("/solve_image", response_model=SolveResponse)
async def solve_image(file: UploadFile = File(...)):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "Falta OPENAI_API_KEY")
    img_bytes = await file.read()
    latex = ask_openai_vision(img_bytes)
    rgba = latex_block_to_rgba(latex)
    pbms = [to_pbm_p4(p) for p in compose_to_pages(rgba)]
    token = uuid.uuid4().hex[:12]
    PAGE_STORE[token] = pbms
    return {"token": token, "pages": len(pbms)}

@app.get("/manifest/{token}")
def manifest(token: str):
    pages = PAGE_STORE.get(token)
    if pages is None: raise HTTPException(404, "Token no encontrado")
    return {"token": token, "pages": len(pages)}

@app.get("/pbm/{token}/{i}")
def get_pbm(token: str, i: int):
    pages = PAGE_STORE.get(token)
    if pages is None or not (0 <= i < len(pages)):
        raise HTTPException(404, "Página no encontrada")
    return Response(content=pages[i], media_type="image/x-portable-bitmap")
