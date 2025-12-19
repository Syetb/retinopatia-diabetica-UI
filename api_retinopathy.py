"""
API para clasificación de Retinopatía Diabética
Compatible con TensorFlow 2.16 + Mac M4 (Apple Silicon)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# CONFIGURACIÓN

CLASS_NAMES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
CLASS_DESCRIPTIONS = {
    'No_DR': 'Sin Retinopatía Diabética',
    'Mild': 'Retinopatía Diabética Leve',
    'Moderate': 'Retinopatía Diabética Moderada',
    'Severe': 'Retinopatía Diabética Severa',
    'Proliferate_DR': 'Retinopatía Diabética Proliferativa'
}
IMG_SIZE = (224, 224)
MODEL_PATH = './transfer_learning/model/vgg16_tl.h5'

# Variable global para el modelo
model = None

# LIFESPAN (reemplaza on_event deprecated en FastAPI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Cargando modelo VGG16...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo cargado exitosamente!")
    print(f"🍎 Ejecutando en: {tf.config.list_physical_devices()}")
    yield
    print("Cerrando API...")

app = FastAPI(
    title="API Retinopatía Diabética",
    description="Clasificación de imágenes retinales usando VGG16",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------
# SCHEMAS


class ImageRequest(BaseModel):
    image_base64: str


class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    description: str
    confidence: float
    all_probabilities: dict

# FUNCIONES


def preprocess_image(image_base64: str) -> np.ndarray:
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    return img_array

# --------------------
# .    ENDPOINTS
# -------------------


@app.get("/")
async def root():
    return {"message": "API Retinopatía Diabética", "status": "ok"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "classes": CLASS_NAMES
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        img_array = preprocess_image(request.image_base64)
        predictions = model.predict(img_array, verbose=0)

        predicted_idx = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx])

        all_probs = {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        return PredictionResponse(
            success=True,
            prediction=predicted_class,
            description=CLASS_DESCRIPTIONS[predicted_class],
            confidence=round(confidence, 4),
            all_probabilities=all_probs
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
