# Analizador de Calorías (Pre-Alpha)

Sistema modular basado en **Deep Learning (CNN + Transfer Learning)** para detectar si una imagen contiene **comida** o **no comida**, y clasificarla posteriormente en múltiples categorías usando un **pipeline en cascada**.

> **Versión Beta disponible:**  
> La versión funcional en producción de este sistema se encuentra desplegada en mi portfolio personal: https://gcodev.es/  
>  
> Esta versión incluye una interfaz interactiva, backend operativo y mejoras respecto a este repositorio, que actúa como **base técnica y experimental** del proyecto.

---
---
##  Resumen

Este proyecto implementa una arquitectura en **cascada de tres capas**:

-  **Filtro binario**: clasificación `food` vs `no_food`
-  **Clasificador de alimentos**: 121 clases (Food-101 ampliado)
-  **Clasificador no-food**: 22 categorías (personas, animales, paisajes, objetos...)

Incluye además un módulo de **estimación nutricional** (calorías y macronutrientes) basado en un CSV nutricional.

Se proporcionan:
- Notebooks reproducibles  
- Scripts de entrenamiento e inferencia  
- Backend FastAPI  
- UI Gradio  
- Docker para despliegue

### Notebooks

Puedes ejecutar y descargar los notebooks desde este repositorio:

- [Demo Analizador de Calorías (Colab-ready)](notebooks/analizador_de_calorías_computer_vision.ipynb)

Ejecutar directamente en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/TU_USUARIO/TU_REPO/blob/main/notebooks/analizador_calorias_computer_vision.ipynb
)

---

## Objetivo

Crear un sistema:
- Reproducible y modular
- Preparado para producción
- Fácilmente extensible (nuevas clases, modelos o fuentes de datos)
- Capaz de integrarse en aplicaciones externas (mobile / web / IoT)

---

## Tabla de contenidos

- [Descripción](#descripción)
- [Objetivo](#objetivo)
- [Arquitectura](#arquitectura)
- [Características](#características)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Entrenamiento](#entrenamiento)
- [Inferencia](#inferencia)
- [API (FastAPI)](#api-fastapi)
- [Interfaz Gradio](#interfaz-gradio)
- [Docker](#docker)
- [Modelos y datos](#modelos-y-datos)
- [Notas técnicas](#notas-técnicas)
- [Estado del proyecto](#estado-del-proyecto)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

---

## Descripción

Este proyecto implementa un **clasificador en cascada** que:

1. Determina si una imagen es **food** o **no_food**
2. Si es *food*, la clasifica entre **hasta 121 tipos de comida**
3. Si es *no_food*, la clasifica en **22 categorías contextuales**
4. Asocia predicciones de comida con **información nutricional estimada**
5. Expone el sistema mediante **API REST (FastAPI)** y **UI (Gradio)**


---

## Arquitectura
```text
Imagen
│
▼
[ Binary Classifier ]
│
├── food ─────▶ [ Food Classifier (121 clases) ] ─▶ Nutrición
│
└── no_food ──▶ [ No-Food Classifier (22 clases) ]
```

### Detalles técnicos
- Backbone: **EfficientNet (timm)**
- Transfer learning + fine-tuning
- Albumentations para data augmentation
- Mixed Precision Training (AMP)
- AdamW + class weighting
- Inferencia con umbral configurable

---

## Diagrama de Flujo


```mermaid
flowchart TD
    INICIO["Inicio train.py"]
    PARSE_ARGS["Parsear argumentos CLI"]
    CHECK_MODE["Validar modo: binary/food/nofood"]
    PREPARE_DATA["Preparar datasets<br/>Food-101 + No-Food"]
    COPY_TMP{"Copy to tmp?<br/>--copy_to_tmp"}
    
    INICIO --> PARSE_ARGS
    PARSE_ARGS --> CHECK_MODE
    CHECK_MODE --> PREPARE_DATA
    
    PREPARE_DATA --> COPY_TMP
    COPY_TMP -->|Sí| COPY_DONE["Copiar dataset a /tmp"]
    COPY_TMP -->|No| LOAD_TRANSFORMS["Cargar transforms<br/>Albumentations"]
    COPY_DONE --> LOAD_TRANSFORMS
    
    LOAD_TRANSFORMS --> CREATE_DATASETS["Crear datasets:<br/>AlbumentationsImageFolder<br/>BinaryFoodNoFoodDataset"]
    CREATE_DATASETS --> COMPUTE_WEIGHTS["Calcular pesos<br/>WeightedRandomSampler"]
    COMPUTE_WEIGHTS --> CREATE_DATALOADERS["DataLoaders:<br/>train/val<br/>workers=2/0 en Colab"]

    subgraph TRAINING_LOOP["🔄 Bucle de Entrenamiento"]
        EPOCH_START["Época N"]
        TRAIN_EPOCH["train_epoch()<br/>AMP + AdamW<br/>CE Loss"]
        EVAL_VAL["eval_model()<br/>Accuracy en val"]
        SAVE_CHECKPOINT{"Mejor acc?<br/>Guardar checkpoint"}
    end

    CREATE_DATALOADERS --> EPOCH_START
    EPOCH_START --> TRAIN_EPOCH
    TRAIN_EPOCH --> EVAL_VAL
    EVAL_VAL --> SAVE_CHECKPOINT
    SAVE_CHECKPOINT --> EPOCH_FIN["Fin época"]
    EPOCH_FIN -.->|Siguiente| EPOCH_START

    EPOCH_FIN --> END_TRAINING["Fin entrenamiento<br/>Best acc reportado"]

    PARSE_ARGS -.->|Error| ERROR_ARGS["Error: faltan argumentos<br/>--mode requerido"]
    CHECK_MODE -.->|Error| ERROR_MODE["Error: modo no soportado"]
    CREATE_DATASETS -.->|Error| ERROR_DATA["Error: dataset vacío"]
    TRAINING_LOOP -.->|Excepción| SAVE_CRASH["crash_partial_{mode}.pth"]

    classDef inicio fill:#a3c1f7,stroke:#333,stroke-width:2px
    classDef proceso fill:#f7efb3,stroke:#333,stroke-width:2px
    classDef decision fill:#ffd5a3,stroke:#333,stroke-width:2px
    classDef error fill:#d9787a,stroke:#a25757,stroke-width:2px
    classDef final fill:#688654,stroke:#333,stroke-width:3px

    class INICIO,EPOCH_START,END_TRAINING inicio
    class PREPARE_DATA,LOAD_TRANSFORMS,CREATE_DATASETS,COMPUTE_WEIGHTS,CREATE_DATALOADERS proceso
    class COPY_TMP,SAVE_CHECKPOINT decision
    class TRAIN_EPOCH,EVAL_VAL final
    class ERROR_ARGS,ERROR_MODE,ERROR_DATA,SAVE_CRASH error
```



---

## Características

- Clasificación en cascada (binario → multiclase)
- Entrenamiento configurable por modo
- Inferencia local o vía API
-  Estimación nutricional desde CSV
-  UI interactiva con Gradio
- ✅ Docker listo para despliegue
- ✅ Compatible con Google Colab

---

## Estructura del repositorio

```text
.
├── app_fastapi.py          # API REST
├── app_gradio.py           # UI Gradio (cliente o local)
├── main.py                 # Demo local Gradio
├── train.py                # Entrenamiento (binary / food / nofood)
├── inference_cascade.py    # Pipeline de inferencia
├── utils.py                # Utilidades comunes
├── models/                 # clases.txt, checkpoints
├── model/                  # modelos .pth
├── data/
│   └── nutrition_food101_merged.csv
├── Dockerfile
├── requirements.txt
└── README.md
```


---

## Requisitos

- Python ≥ 3.9
- PyTorch
- GPU recomendada (para entrenamiento)

Principales librerías:
- `torch`, `timm`
- `albumentations`
- `fastapi`, `uvicorn`
- `gradio`
- `pandas`, `numpy`, `scikit-learn`

---

## Instalación

```bash
pip install -r requirements.txt
```

Instalación manual mínima:
```
pip install timm==0.9.2 albumentations==1.3.0 torchmetrics scikit-learn
pip install fastapi uvicorn gradio pandas numpy
```
---

---

## Entrenamiento

El script `train.py` soporta tres modos:

- `binary`
- `food`
- `nofood`

### Ejemplo (binario)

```bash
python train.py \
  --mode binary \
  --data_dir /path/Food-101 \
  --no_food_dir /path/no_food \
  --model_dir ./models \
  --epochs 10 \
  --bs 32 \
  --img_size 192

```

### Diagrama de Flujo del Entrenamiento

text
## 🏋️ Diagrama de Flujo del Entrenamiento (Sin Errores)

```mermaid
flowchart TD
    INICIO["Inicio train.py"]
    PARSE_ARGS["Parsear argumentos CLI"]
    CHECK_MODE["Validar modo"]
    PREPARE_DATA["Preparar datasets Food-101"]
    COPY_TMP{"Copy to tmp?"}
    
    INICIO --> PARSE_ARGS
    PARSE_ARGS --> CHECK_MODE
    CHECK_MODE --> PREPARE_DATA
    
    PREPARE_DATA --> COPY_TMP
    COPY_TMP -->|Sí| COPY_DONE["Copiar a /tmp"]
    COPY_TMP -->|No| LOAD_TRANSFORMS["Cargar Albumentations"]
    COPY_DONE --> LOAD_TRANSFORMS
    
    LOAD_TRANSFORMS --> CREATE_DATASETS["Datasets train/val"]
    CREATE_DATASETS --> COMPUTE_WEIGHTS["WeightedRandomSampler"]
    COMPUTE_WEIGHTS --> CREATE_DATALOADERS["DataLoaders"]

    subgraph TRAINING_LOOP["Bucle de Entrenamiento"]
        EPOCH_START["Epoca N"]
        TRAIN_EPOCH["train_epoch AMP+AdamW"]
        EVAL_VAL["eval_model val acc"]
        SAVE_CHECKPOINT["Guardar best_model"]
    end

    CREATE_DATALOADERS --> EPOCH_START
    EPOCH_START --> TRAIN_EPOCH
    TRAIN_EPOCH --> EVAL_VAL
    EVAL_VAL --> SAVE_CHECKPOINT
    SAVE_CHECKPOINT --> EPOCH_FIN["Fin epoca"]
    EPOCH_FIN -.->|Next| EPOCH_START
    EPOCH_FIN --> END_TRAINING["Best acc guardado"]

    classDef inicio fill:#a3c1f7
    classDef proceso fill:#f7efb3
    classDef decision fill:#ffd5a3
    classDef loop fill:#c9e4a1

    class INICIO,END_TRAINING inicio
    class PREPARE_DATA,LOAD_TRANSFORMS,CREATE_DATASETS,COMPUTE_WEIGHTS,CREATE_DATALOADERS proceso
    class COPY_TMP,SAVE_CHECKPOINT decision
    class EPOCH_START,EPOCH_FIN loop
```

---

##  Inferencia

```from inference_cascade import predict_single

result = predict_single("image.jpg", bin_thresh=0.5)
print(result)
```
Salida típica: (food, "pizza", 0.94)

---

## Entrenamiento

El script `train.py` soporta tres modos:

- `binary`
- `food`
- `nofood`

### Ejemplo (binario)

```bash
python train.py \
  --mode binary \
  --data_dir /path/Food-101 \
  --no_food_dir /path/no_food \
  --model_dir ./models \
  --epochs 10 \
  --bs 32 \
  --img_size 192
Los modelos y archivos de clases se guardan automáticamente en --model_dir.
```

---

## Inferencia
Inferencia en cascada sobre una imagen:
```
from inference_cascade import predict_single

result = predict_single("image.jpg", bin_thresh=0.5)
print(result)
```
Salida típica:
```
(food, "pizza", 0.94)
```
---

## API (FastAPI)
Lanzar servidor
```
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
Endpoint principal
POST /predict

curl -X POST "http://localhost:8000/predict?topk=3" \
  -F "file=@image.jpg"
```
Devuelve:
```
Predicción top-k
```
Probabilidades

Información nutricional estimada (si aplica)

---

## Interfaz Gradio
python app_gradio.py
# 
python main.py
Opcionalmente, puede consumir la API remota configurando:

export BACKEND_URL="http://localhost:8000"

---

## Docker

Construir imagen

docker build -t food-classifier .
Ejecutar
docker run -p 8000:8000 food-classifier
Puedes montar volúmenes para modelos y datos si lo prefieres.

## Modelos y datos

- Food: Food-101 + platos adicionales (121 clases)
- No-food: 22 categorías
- Nutrición: CSV con calorías y macronutrientes estimados

> Asegúrate de que `classes_*.txt` coincidan exactamente con los checkpoints usados.

---

## Notas técnicas

- Copiar datasets desde Google Drive a disco local mejora significativamente el rendimiento
- En Colab, usar `workers=0` si hay bloqueos del `DataLoader`
- Los checkpoints pueden requerir limpieza de prefijos (`module.`)
- El proyecto utiliza label smoothing por defecto

---

##  Estado del proyecto

- 🟡 Pre-alfa
- Código funcional
- Falta hardening para producción (tests, validaciones, seguridad)
- Ideal para investigación, demos y prototipos

---

## Contribuir

¡Las contribuciones son bienvenidas! Si quieres mejorar este proyecto:

1. Abre un **issue** para proponer cambios, reportar bugs o sugerir mejoras.
2. Crea una nueva rama a partir de `main`:
   - `feature/nombre-funcionalidad`
   - `fix/descripcion-bug`
3. Realiza tus cambios asegurándote de que:
   - El código sea claro y esté documentado
   - No rompa funcionalidades existentes
4. Envía un **Pull Request** describiendo claramente:
   - Qué se ha cambiado
   - Por qué es necesario
   - Cómo probarlo

> Sugerencia: si el cambio es grande, abre primero un issue para discutirlo.

---

## 📄 Licencia

Este proyecto se publicará bajo licencia **MIT**, lo que permite su uso, modificación y distribución libremente, siempre que se mantenga la atribución al autor.












