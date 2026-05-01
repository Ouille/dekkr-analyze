"""
DekkR Analyze — Service d'analyse audio (Essentia)
Licence : AGPL-3.0
"""

import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import tempfile

from analysis import analyze_audio

API_KEY = os.environ.get("API_KEY", "")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 Mo


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pré-charger les algorithmes Essentia au démarrage pour éviter
    # que la première vraie requête soit lente et retourne un 502 timeout.
    try:
        import essentia.standard as es
        dummy = np.random.randn(22050).astype(np.float32) * 0.01
        es.KeyExtractor(sampleRate=22050)(dummy)
        es.RhythmExtractor2013(method="multifeature", sampleRate=22050)(dummy)
        print("[startup] Essentia warmup OK")
    except Exception as e:
        print(f"[startup] Essentia warmup skip: {e}")
    yield


app = FastAPI(
    title="DekkR Analyze",
    description="Service d'analyse audio (tonalité, BPM, structure) via Essentia",
    version="1.0.0",
    license_info={"name": "AGPL-3.0"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()

SUPPORTED_FORMATS = {
    "audio/mpeg", "audio/wav", "audio/flac",
    "audio/aac", "audio/ogg", "audio/x-flac",
}


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEY or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")
    return credentials.credentials


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    _key: str = Depends(verify_api_key),
):
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=422, detail="Fichier trop volumineux (max 50 Mo)")

    # Écrire dans un fichier temporaire pour Essentia
    suffix = "." + (file.filename or "audio.mp3").rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analyze_audio(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse : {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}
