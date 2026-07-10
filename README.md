# Taller B4-T2 · XAI aplicado a concesión de crédito

Práctica del Máster MIAX (Bloque 4) sobre explicabilidad de modelos de machine learning (XAI).

## Descripción

El objetivo de la práctica es **construir, auditar y optimizar un modelo de concesión de crédito**
bajo dos escenarios de coste distintos, y justificar las decisiones de negocio con técnicas de
explicabilidad (XAI).

## Grupo

- Nombre 1 (apellidos, email)
- Nombre 2 (apellidos, email)
- Nombre 3 (apellidos, email)

## Objetivo

Se debe entrenar y optimizar un modelo de scoring de crédito en dos condiciones de coste:

| Escenario | Coste Falso Positivo | Coste Falso Negativo | Entregable |
|---|---|---|---|
| 1 | 1 | 1 | `results/predicciones/cs_produccion1.csv` |
| 2 | 10 | 10 | `results/predicciones/cs_produccion2.csv` |

Los modelos candidatos son: a) modelo supervisado, b) Multiarmed Bandit.

## Auditoría del modelo

Además de construir y optimizar el modelo, se debe auditar mediante:

- **Modelo subrogado** con reglas que expliquen el comportamiento del modelo principal.
- **Contrafactuales** para varios ejemplos seleccionados de clase real 0 y 1 (¿qué información
  le damos a un cliente que pide explicaciones sobre por qué se le deniega un crédito?).
- **SHAP global y local** sobre las predicciones del modelo.
- Otras técnicas de explicabilidad que el grupo considere oportunas.

## Estructura del repositorio

```
├── docs/                  # Documentación de teoría, fuentes y decisiones de proyecto
│   ├── INDEX.md           # Índice de la documentación
│   ├── DECISIONES.md      # Registro de decisiones (ADR-like) del proyecto
│   ├── _fuentes/          # Material de clase y fuentes originales
│   └── teoria/            # Notas de teoría (SHAP, contrafactuales, subrogados, etc.)
├── data/                  # Datos del taller (no versionados, ver .gitignore)
├── notebooks/             # Notebooks de trabajo 01-07 y notebook de entrega 99_ENTREGA.ipynb
├── src/                   # Código fuente reutilizable
│   ├── preprocessing.py   # Pipeline de preprocesado fit/transform (IMPLEMENTADO)
│   ├── modeling.py        # Entrenamiento y evaluación de modelos (stub, pendiente)
│   ├── cost_utils.py      # Matrices de coste, coste esperado, optimización de umbral (stub, pendiente)
│   └── xai_utils.py       # Wrappers de SHAP, modelo subrogado y contrafactuales (stub, pendiente)
├── data/
│   └── processed/         # Salidas del preprocesado: train/test/produccion.parquet (generados)
├── results/               # Resultados generados
│   ├── figures/           # Gráficas (incluye prep_*.png del preprocesado)
│   ├── tables/            # Tablas de resultados (incluye prep_*.csv del preprocesado)
│   ├── models/            # Artefactos serializados (preprocessing_pipeline.joblib)
│   └── predicciones/      # cs_produccion1.csv y cs_produccion2.csv (entregables, sí versionados)
└── report/                # Material auxiliar para el informe/entrega final
```

### Convención de notebooks

- `01`–`07`: notebooks de trabajo, exploración y desarrollo incremental, uno por bloque temático:
  `01` EDA, `02` preprocesado, `03` modelo coste 1, `04` modelo coste 10, `05` auditoría subrogado,
  `06` contrafactuales, `07` SHAP.
- `99_ENTREGA.ipynb`: notebook único consolidado que se entrega, con el código final, la
  justificación de los desarrollos, las tablas de resultados y la reflexión final.

## Estado actual

- **Documentación**: síntesis de teoría completa en `docs/teoria/` (SHAP, contrafactuales,
  subrogados, coste/umbral, bandits, notebook de partida del profesor), citando siempre la fuente
  en `docs/_fuentes/`. Decisiones de diseño registradas en `docs/DECISIONES.md`: D-0.1 a D-0.3
  siguen ABIERTAS a la espera de evidencia empírica del modelado real (notebooks `03`/`04`); el
  bloque de preprocesado D-1.1 a D-1.6 (imputación, outliers/recálculo de DebtRatio, `age == 0`,
  split, escalado y duplicados) quedó CERRADO tras implementar el `02`.
- **Notebooks**: `01_EDA.ipynb` y `02_preprocesado.ipynb` están completos y ejecutados de principio
  a fin sin errores. El `01` tiene el análisis exploratorio (carga/validación, EDA básico, valores
  centinela y outliers, EDA avanzado con comparación train/producción, correlaciones, PCA lineal y
  no lineal, recomendaciones de nulos y normalización, conclusiones). El `02` construye el pipeline
  de preprocesado (`src/preprocessing.py`) y deja sus artefactos: `data/processed/train.parquet`,
  `test.parquet` y `produccion.parquet`, más `results/models/preprocessing_pipeline.joblib` (y
  tablas `prep_*.csv` / figuras `prep_*.png`). Los notebooks `03`–`07` y `99_ENTREGA.ipynb` siguen
  siendo esqueletos con celdas `# TODO` (estructura, conectores y decisiones ya definidos, código
  pendiente de implementar).

## Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows; en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Para trabajar de forma interactiva:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

Para ejecutar un notebook de principio a fin desde la terminal, regenerando sus resultados en
`results/`. **Atención a la convención de rutas, que difiere entre notebooks:**

- `01_EDA.ipynb` asume `cwd = notebooks/` y lee los datos con rutas `../data/...`. Se ejecuta
  entrando en la carpeta:

  ```bash
  cd notebooks
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 01_EDA.ipynb
  ```

- `02_preprocesado.ipynb` asume `cwd = raíz del repo` y usa rutas `data/...`, `results/...` (sin
  `../`). Incluye una guarda que hace `os.chdir("..")` si el kernel arranca en `notebooks/`, así que
  se puede lanzar directamente desde la raíz:

  ```bash
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/02_preprocesado.ipynb
  ```

Los datos reales del taller (`cs_construccion.csv`, `cs_produccion.csv`, `DataDictionary.csv`) se
copian a `data/` desde `docs/_fuentes/` (no versionados, ver `.gitignore`).
