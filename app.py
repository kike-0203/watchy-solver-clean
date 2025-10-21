from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64, hashlib, os, textwrap, tempfile
from openai import OpenAI

OPENAI_MODEL = "gpt-4o-mini"
PROMPT_ES = (
    "Analiza cuidadosamente el texto o ecuación en la imagen y responde "
    "explicando la solución de manera breve y en formato matemático LaTeX, "
    "sin texto adicional ni comentarios. Si es una ecuación diferencial, resuélvela paso a paso."
)

app = FastAPI()

TMP = tempfile.gettempdir()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- nueva función corregida ---
def ask_openai_vision(image_bytes: bytes) -> str:
    """
    Envía la imagen como data URL (base64) para evitar 'bytes no serializable'.
    Intenta primero la Responses API (SDK nuevo) y cae a Chat Completions si falla.
    """
    import base64
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    # Ruta 1: SDK nuevo (Responses API)
    try:
        msg = [
            {"role": "user", "content": [
                {"type": "input_text", "text": PROMPT_ES},
                {"type": "input_image", "image_url": {"url": data_url}}
            ]}
        ]
        res = client.responses.create(model=OPENAI_MODEL, input=msg)
        if hasattr(res, "output_text"):
            return res.output_text.strip()
        try:
            return res.output[0].content[0].text.strip()
        except Exception:
            pass
    except Exception:
        pass

    # Ruta 2: fallback a Chat Completions multimodal
    import openai as openai_legacy
    openai_legacy.api_key = os.getenv("OPENAI_API_KEY")
    r = openai_legacy.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_ES},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }]
    )
    return r.choices[0].message["content"].strip()


def latex_block_to_rgba(text: str, page_w=200, page_h=200) -> list[Image.Image]:
    """
    Convierte el texto de salida (LaTeX plano o normal) a páginas 200x200 píxeles PBM.
    """
    lines = textwrap.wrap(text, width=24)
    per_page = page_h // 14
    pages = []
    for i in range(0, len(lines), per_page):
        img = Image.new("1", (page_w, page_h), 1)
        d = ImageDraw.Draw(img)
        y = 10
        for line in lines[i:i + per_page]:
            d.text((10, y), line, fill=0)
            y += 14
        pages.append(img)
    return pages


@app.post("/solve_image")
async def solve_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    latex = ask_openai_vision(img_bytes)
    token = hashlib.sha1(img_bytes).hexdigest()[:12]
    pages = latex_block_to_rgba(latex)
    os.makedirs(os.path.join(TMP, token), exist_ok=True)
    for i, p in enumerate(pages):
        p.save(os.path.join(TMP, token, f"{i}.pbm"))
    return JSONResponse({"token": token, "pages": len(pages)})


@app.get("/pbm/{token}/{page}")
async def get_pbm(token: str, page: int):
    path = os.path.join(TMP, token, f"{page}.pbm")
    if not os.path.exists(path):
        return JSONResponse({"error": "No existe esa página"}, status_code=404)
    return Response(content=open(path, "rb").read(), media_type="image/x-portable-bitmap")

