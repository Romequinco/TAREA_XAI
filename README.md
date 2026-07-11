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

> **Nota de interpretación.** El enunciado escribe ambos escenarios como simétricos (FP = FN). Tras
> analizar el material de partida, el escenario 2 se interpreta como coste **asimétrico**
> C_FN = 10·C_FP (denegar es más barato que conceder a un moroso); ver la decisión D-3.2 en
> `docs/DECISIONES.md`. La lectura literal (FP = FN = 10) queda registrada como respaldo en
> `results/predicciones/cs_produccion2_literal.csv`.

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
│       (docs/_fuentes/ NO se versiona: material bruto privado del profesor)
├── data/                  # Datos del taller: CSV de entrada + processed/ (versionados)
├── notebooks/             # Notebooks de trabajo 01-07 y notebook de entrega 99_ENTREGA.ipynb
├── src/                   # Código fuente reutilizable
│   ├── preprocessing.py   # Pipeline de preprocesado fit/transform (IMPLEMENTADO)
│   ├── modeling.py        # Familias de modelos (lineal/boosted/neuronal/bandit) y comparación por CV (IMPLEMENTADO)
│   ├── cost_utils.py      # Matrices de coste, coste esperado, optimización de umbral (IMPLEMENTADO)
│   └── xai_utils.py       # Wrappers de SHAP, modelo subrogado y contrafactuales (stub, pendiente)
├── data/
│   └── processed/         # Salidas del preprocesado: train/test/produccion.parquet (generados)
├── results/               # Resultados generados
│   ├── figures/           # Gráficas (incluye prep_*.png del preprocesado)
│   ├── tables/            # Tablas de resultados (incluye prep_*.csv del preprocesado)
│   ├── models/            # Artefactos serializados (preprocessing_pipeline.joblib)
│   └── predicciones/      # cs_produccion1.csv y cs_produccion2.csv (entregables); cs_produccion2_literal.csv (respaldo lectura literal)
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
  en `docs/_fuentes/`. Decisiones de diseño registradas en `docs/DECISIONES.md`: el bloque de
  preprocesado D-1.1 a D-1.6 (imputación, outliers/recálculo de DebtRatio, `age == 0`, split,
  escalado y duplicados) quedó CERRADO tras implementar el `02`; y el 2026-07-11, con la evidencia
  empírica del modelado real, se cerraron también D-0.1 (modelo supervisado, no bandit), D-0.2 (un
  único modelo con dos umbrales), D-0.3 (XGBoost), D-2.1 (validación cruzada anti-fuga), D-2.2
  (modelo de producción), D-3.1 (método del umbral) y D-3.2 (interpretación asimétrica del escenario
  2). **Ya no queda ninguna decisión de modelo o coste abierta.**
- **Notebooks**: `01_EDA.ipynb`, `02_preprocesado.ipynb`, `03_modelo_coste1.ipynb` y
  `04_modelo_coste10.ipynb` están completos y ejecutados de principio a fin sin errores. El `01`
  tiene el análisis exploratorio (carga/validación, EDA básico, valores centinela y outliers, EDA
  avanzado con comparación train/producción, correlaciones, PCA lineal y no lineal, recomendaciones
  de nulos y normalización, conclusiones). El `02` construye el pipeline de preprocesado
  (`src/preprocessing.py`) y deja sus artefactos: `data/processed/train.parquet`, `test.parquet` y
  `produccion.parquet`, más `results/models/preprocessing_pipeline.joblib` (y tablas `prep_*.csv` /
  figuras `prep_*.png`). El `03` (coste simétrico) y el `04` (coste asimétrico) comparan por
  validación cruzada anti-fuga varias familias de modelos (logística, XGBoost, dos MLP Keras y un
  bandit LinUCB propio) y eligen el umbral por coste esperado; XGBoost gana en ambos escenarios.
  Generan los dos entregables `results/predicciones/cs_produccion1.csv` y `cs_produccion2.csv`
  (más el respaldo `cs_produccion2_literal.csv`), los modelos `modelo_coste1.joblib` /
  `modelo_coste10.joblib`, la tabla `results/tables/umbrales_coste.csv` y las tablas/figuras
  `mod_03_*` / `mod_04_*`. Los notebooks `05`–`07` y `99_ENTREGA.ipynb` siguen siendo esqueletos con
  celdas `# TODO` (estructura, conectores y decisiones ya definidos, código pendiente de
  implementar).

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

### Repositorio plug-and-play

El repositorio se versiona de forma que **cualquiera puede clonarlo y ejecutar cualquier notebook
sin regenerar la cadena anterior**. SÍ están versionados: los datos de entrada (`data/*.csv`), los
intermedios del preprocesado (`data/processed/*.parquet`), los modelos (`results/models/*.joblib`),
y las predicciones, tablas y figuras de `results/`. Así que basta con:

```bash
git clone <repo> && cd TAREA_XAI
python -m venv .venv && .venv\Scripts\activate   # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/05_auditoria_subrogado.ipynb   # p. ej., sin haber ejecutado 02-04
```

Lo ÚNICO que NO se versiona es `docs/_fuentes/`: el material bruto privado del profesor (enunciado
en PDF, transparencias, notebooks de ejemplo y los datasets originales). Quien lo necesite debe
conseguirlo aparte; los notebooks **no dependen de esa carpeta** para ejecutarse (leen de `data/`).
El `.gitignore` está configurado para que los artefactos de los notebooks pendientes (`05`, `06`,
`07`, `99`) también queden versionados automáticamente al generarse.
