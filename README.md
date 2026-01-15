# 🩺 Detección Temprana de Retinopatía Diabética mediante Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.12.0-red.svg)
![VGG16](https://img.shields.io/badge/Architecture-VGG16-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*Modelo de clasificación automática de imágenes retinales usando Transfer Learning y Fine-Tuning*


</div>

---

## 📋 Descripción

Este proyecto implementa un **modelo de clasificación multiclase** para la detección temprana de la retinopatía diabética a partir de imágenes de fondo de ojo. Utilizando técnicas avanzadas de **Deep Learning**, específicamente **Transfer Learning** con la arquitectura **VGG16** pre-entrenada en ImageNet, el modelo es capaz de clasificar automáticamente imágenes retinales en cinco categorías clínicas.

### 🎯 Objetivo Principal

Desarrollar y evaluar un modelo de clasificación basado en aprendizaje profundo que permita la detección temprana y precisa de diferentes grados de retinopatía diabética, contribuyendo así a la prevención de la ceguera en pacientes diabéticos.

---

## 🔬 Categorías de Clasificación

El modelo clasifica las imágenes retinales en las siguientes categorías:

| Categoría | Descripción |
|-----------|-------------|
| **No_DR** | Sin retinopatía diabética |
| **Mild** | Retinopatía diabética leve |
| **Moderate** | Retinopatía diabética moderada |
| **Severe** | Retinopatía diabética severa |
| **Proliferate_DR** | Retinopatía diabética proliferativa |

---

## ✨ Características

### 🏗️ Arquitectura del Modelo

- **Base**: VGG16 pre-entrenada en ImageNet
- **Técnica**: Transfer Learning con capas convolucionales congeladas
- **Fine-Tuning**: Ajuste de capas superiores para especialización en imágenes retinales
- **Cabeza personalizada**: Clasificador adaptado para 5 clases

### 🔧 Procesamiento de Datos

- **Redimensionamiento**: 224×224 píxeles (estándar VGG16)
- **Normalización**: Preprocesamiento según esquema ImageNet
- **Data Augmentation**: 
  - Rotaciones aleatorias
  - Desplazamientos horizontales y verticales
  - Zoom dinámico
  - Volteos horizontales
- **División de datos**: 80% entrenamiento / 20% validación

### 📊 Evaluación y Explicabilidad

- **Métricas**: Accuracy, Precision (macro), Recall (macro), F1-Score (macro)
- **Análisis por clase**: Matrices de confusión detalladas
- **Interpretabilidad**: Grad-CAM para visualización de regiones de interés
- **Validación clínica**: Verificación de enfoque en estructuras relevantes

---

## 📦 Dataset

**Fuente**: [Diabetic Retinopathy Dataset - Mendeley Data](https://data.mendeley.com/) (Tuna, 2025)

El dataset contiene imágenes de fondo de ojo organizadas por categoría clínica, permitiendo el entrenamiento supervisado del modelo.

### Estructura del Dataset
```
dataset/
├── Mild/
│   ├── imagen_001.jpg
│   ├── imagen_002.jpg
│   └── ...
├── Moderate/
│   ├── imagen_001.jpg
│   └── ...
├── No_DR/
│   ├── imagen_001.jpg
│   └── ...
├── Proliferate_DR/
│   ├── imagen_001.jpg
│   └── ...
└── Severe/
    ├── imagen_001.jpg
    └── ...
```
---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 8GB+ de RAM recomendado
- GPU con soporte CUDA (opcional, pero recomendado para entrenamiento)

### Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/retinopatia-diabetica-dl.git
cd retinopatia-diabetica-dl


### Instalación de Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

```
tensorflow==2.16.2
keras==3.12.0
numpy==1.26.4
pandas==2.3.3
matplotlib==3.10.8
scikit-learn==1.7.2
seaborn==0.13.2
tf-keras-vis==0.8.2
pillow==12.0.0
streamlit==1.52.2
```

---

## 💻 Uso

### 1. Preparación del Dataset

Descarga el dataset y organízalo en la estructura de carpetas mencionada anteriormente. Cada carpeta debe contener las imágenes correspondientes a su categoría clínica.

```python
# Configurar la ruta del dataset
DATA_PATH = './dataset'

# Verificar estructura
import os
classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
for class_name in classes:
    path = os.path.join(DATA_PATH, class_name)
    print(f"{class_name}: {len(os.listdir(path))} imágenes")
```

### 2. Preprocesamiento de Datos

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Data Augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)

# Generador de validación
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT
)
```

### 3. Entrenamiento del Modelo

```python
# Entrenar modelo con Transfer Learning
python train.py --data_path ./dataset --epochs 50 --batch_size 32 --lr 0.0001

# Entrenar con Fine-Tuning
python train.py --data_path ./dataset --epochs 30 --batch_size 32 --lr 0.00001 --fine_tune
```

**Parámetros disponibles:**
- `--data_path`: Ruta al dataset
- `--epochs`: Número de épocas de entrenamiento
- `--batch_size`: Tamaño del batch
- `--lr`: Tasa de aprendizaje
- `--fine_tune`: Activar fine-tuning

### 4. Evaluación del Modelo

```python
# Evaluar modelo en conjunto de validación
python evaluate.py --model_path ./models/best_model.h5 --data_path ./dataset

# Generar matriz de confusión
python evaluate.py --model_path ./models/best_model.h5 --data_path ./dataset --confusion_matrix
```

### 5. Predicción en Nuevas Imágenes

```python
# Predecir una sola imagen
python predict.py --model_path ./models/best_model.h5 --image_path ./test_images/retina.jpg

# Predecir múltiples imágenes
python predict.py --model_path ./models/best_model.h5 --images_dir ./test_images/
```

### 6. Visualización con Grad-CAM

```python
# Generar mapas de calor Grad-CAM
python gradcam.py --model_path ./models/best_model.h5 --image_path ./test_images/retina.jpg --output_path ./gradcam_results/
```

### 7. Interfaz Interactiva con Streamlit

```bash
# Ejecutar aplicación web
streamlit run app.py
```

La aplicación permite cargar imágenes, obtener predicciones en tiempo real y visualizar los mapas de calor Grad-CAM.

---

## 📈 Resultados

### Comparación de Modelos

| Modelo | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|--------|----------|-------------------|----------------|------------------|
| **Transfer Learning + Fine-Tuning (VGG16)** | **XX.X%** | **XX.X%** | **XX.X%** | **XX.X%** |
| Modelo entrenado desde cero | XX.X% | XX.X% | XX.X% | XX.X% |

### Ventajas del Transfer Learning

✅ **Mayor exactitud global** - Supera al modelo entrenado desde cero en todas las métricas  
✅ **Convergencia más rápida** - Requiere menos épocas para alcanzar el óptimo  
✅ **Mayor estabilidad** - Menor varianza en los resultados de validación  
✅ **Mejor generalización** - Aprovecha características aprendidas de ImageNet  
✅ **Eficiencia en datos** - Obtiene buenos resultados con menos datos de entrenamiento

### Rendimiento por Clase

| Clase | Precision | Recall | F1-Score | Muestras |
|-------|-----------|--------|----------|----------|
| No_DR | XX.X% | XX.X% | XX.X% | XXX |
| Mild | XX.X% | XX.X% | XX.X% | XXX |
| Moderate | XX.X% | XX.X% | XX.X% | XXX |
| Severe | XX.X% | XX.X% | XX.X% | XXX |
| Proliferate_DR | XX.X% | XX.X% | XX.X% | XXX |

### Explicabilidad con Grad-CAM

El modelo utiliza **Gradient-weighted Class Activation Mapping (Grad-CAM)** para visualizar las regiones de las imágenes que más influyen en las decisiones de clasificación. Los análisis demuestran que el modelo se enfoca correctamente en:

- 🔴 Microaneurismas
- 🔴 Hemorragias retinales
- 🔴 Exudados duros y blandos
- 🔴 Neovascularización
- 🔴 Alteraciones en la red vascular

Esto valida que el clasificador identifica estructuras **clínicamente relevantes** para el diagnóstico de retinopatía diabética.

---

## 🛠️ Metodología

### 1. Revisión del Estado del Arte

Se realizó un análisis exhaustivo de las técnicas actuales de Deep Learning aplicadas a la detección de retinopatía diabética, identificando:

- Arquitecturas más efectivas (VGG, ResNet, Inception, EfficientNet)
- Técnicas de Transfer Learning y Fine-Tuning
- Métodos de preprocesamiento específicos para imágenes retinales
- Estrategias de explicabilidad en modelos médicos

### 2. Preparación de Datos

**Organización del Dataset:**
- Estructura jerárquica por categorías clínicas
- Verificación de integridad de imágenes
- Balance de clases mediante data augmentation

**Preprocesamiento:**
```python
1. Redimensionamiento → 224×224 píxeles
2. Normalización → preprocess_input (ImageNet)
3. Data Augmentation → Rotación, zoom, flip, shift
4. División → 80% train / 20% validation
```

### 3. Implementación del Modelo

**Fase 1: Transfer Learning**
```python
# Cargar VGG16 pre-entrenada
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelar capas convolucionales
for layer in base_model.layers:
    layer.trainable = False

# Agregar cabeza de clasificación
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(5, activation='softmax')(x)
```

**Fase 2: Fine-Tuning**
```python
# Descongelar últimas capas
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompilar con learning rate menor
optimizer = Adam(learning_rate=1e-5)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 4. Entrenamiento

**Hiperparámetros optimizados:**
- Learning rate (Transfer Learning): 1e-4
- Learning rate (Fine-Tuning): 1e-5
- Batch size: 32
- Épocas Transfer Learning: 50
- Épocas Fine-Tuning: 30
- Optimizer: Adam
- Loss: Categorical Crossentropy

**Callbacks utilizados:**
- ModelCheckpoint: Guardar mejor modelo
- EarlyStopping: Prevenir sobreajuste
- ReduceLROnPlateau: Ajustar learning rate dinámicamente

### 5. Evaluación y Validación

**Métricas de clasificación:**
```python
- Accuracy global
- Precision macro
- Recall macro
- F1-Score macro
- Matriz de confusión
- Curvas ROC por clase
- Análisis de errores
```

**Validación clínica:**
- Revisión de predicciones incorrectas
- Análisis de casos límite entre clases
- Comparación con criterios diagnósticos establecidos

### 6. Interpretabilidad

**Implementación de Grad-CAM:**
```python
# Generar mapas de activación
def generate_gradcam(model, image, class_idx):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer('block5_conv3').output, 
                 model.output]
    )
    # Calcular gradientes y generar heatmap
    ...
```

**Beneficios:**
- Transparencia en las decisiones del modelo
- Validación de que el modelo identifica regiones relevantes
- Detección de posibles sesgos o errores sistemáticos
- Mayor confianza para uso clínico

---

## 📁 Estructura del Proyecto

```
retinopatia-diabetica-dl/
│
├── data/                           # Dataset y archivos de datos
│   ├── raw/                        # Imágenes originales
│   ├── processed/                  # Imágenes preprocesadas
│   └── splits/                     # División train/val/test
│
├── models/                         # Modelos entrenados
│   ├── transfer_learning/          # Modelos con Transfer Learning
│   ├── fine_tuned/                 # Modelos con Fine-Tuning
│   └── checkpoints/                # Checkpoints durante entrenamiento
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_fine_tuning.ipynb
│   └── 05_evaluation_gradcam.ipynb
│
├── src/                           # Código fuente
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # Preprocesamiento de imágenes
│   │   └── data_generator.py      # Generadores de datos
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vgg16_model.py         # Arquitectura VGG16
│   │   ├── train.py               # Lógica de entrenamiento
│   │   └── evaluate.py            # Evaluación del modelo
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── gradcam.py             # Implementación Grad-CAM
│   │   └── plots.py               # Gráficos y visualizaciones
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Configuraciones globales
│       └── helpers.py             # Funciones auxiliares
│
├── scripts/                       # Scripts ejecutables
│   ├── train.py                   # Script de entrenamiento
│   ├── evaluate.py                # Script de evaluación
│   ├── predict.py                 # Script de predicción
│   └── gradcam.py                 # Generar visualizaciones
│
├── results/                       # Resultados y métricas
│   ├── metrics/                   # Métricas de evaluación
│   ├── confusion_matrices/        # Matrices de confusión
│   ├── gradcam_visualizations/    # Visualizaciones Grad-CAM
│   └── reports/                   # Reportes generados
│
├── tests/                         # Tests unitarios
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_gradcam.py
│
├── app.py                         # Aplicación Streamlit
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Este archivo
├── LICENSE                        # Licencia del proyecto
└── .gitignore                     # Archivos ignorados por git
```


### ⭐ Si este proyecto te resulta útil, considera darle una estrella ⭐

**Hecho con ❤️ para mejorar el diagnóstico de retinopatía diabética**

</div>
