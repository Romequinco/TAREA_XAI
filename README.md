# Taller B4-T2 · XAI aplicado a concesión de crédito

Práctica del Máster MIAX (Bloque 4) sobre explicabilidad de modelos de machine learning (XAI).

## Descripción

El objetivo de la práctica es **construir, auditar y optimizar un modelo de concesión de crédito**
bajo dos escenarios de coste distintos, y justificar las decisiones de negocio con técnicas de
explicabilidad (XAI).

La entrega consta del notebook único **`99_ENTREGA.ipynb`** (autocontenido, con el código real que
genera cada resultado) y de un informe de entrega disponible en **tres formatos**: el informe visual
autocontenido (`report/informe_visual.html` y su versión `report/informe_visual.pdf`) y el informe
técnico-académico en LaTeX (`report/Informe_Tecnico_XAI_Credito.tex`, con sus figuras en
`report/figuras/`), cuyo PDF compilado está en la raíz del repositorio como
`Informe_Tecnico_XAI_Credito.pdf` para acceso directo.

## Grupo

- Oscar Romero Quincoces
- Fernando Dapena Tauste
- Daniel Garcia Lopez

## Objetivo

Se debe entrenar y optimizar un modelo de scoring de crédito en dos condiciones de coste:

| Escenario | Coste Falso Positivo | Coste Falso Negativo | Entregable |
|---|---|---|---|
| 1 | 1 | 1 | `results/predicciones/cs_produccion1.csv` |
| 2 | 1 | 10 | `results/predicciones/cs_produccion2.csv` |

> **Nota sobre el escenario 2.** La tabla del enunciado escribía ambos escenarios como simétricos
> (FP = FN), pero **el profesor confirmó que era una errata**: el escenario 2 es coste **asimétrico**
> C_FN = 10·C_FP (conceder a un moroso es 10× más caro que denegar a un buen cliente). Así se
> implementa, de forma definitiva y sin ambigüedad en el código; ver la decisión D-3.2 en
> `docs/DECISIONES.md`.

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
├── Informe_Tecnico_XAI_Credito.pdf  # PDF del informe técnico-académico (en la raíz, junto al README)
├── docs/                  # Documentación de teoría, fuentes y decisiones de proyecto
│   ├── INDEX.md           # Índice de la documentación
│   ├── DECISIONES.md      # Registro de decisiones (ADR-like) del proyecto
│   ├── taller_XAI.pdf     # Enunciado del taller (referencia del proyecto)
│   ├── _fuentes/          # Material de clase y fuentes originales
│   └── teoria/            # Notas de teoría (SHAP, contrafactuales, subrogados, etc.)
│       (docs/_fuentes/ NO se versiona: material bruto privado del profesor)
├── data/                  # Datos del taller: CSV de entrada + processed/ (versionados)
│   └── processed/         # Salidas del preprocesado: train/test/produccion.parquet (generados)
├── notebooks/             # Notebooks de trabajo 01-08 y notebook de entrega 99_ENTREGA.ipynb
├── src/                   # Código fuente reutilizable
│   ├── preprocessing.py   # Pipeline de preprocesado fit/transform (IMPLEMENTADO)
│   ├── modeling.py        # Familias de modelos (lineal/boosted/neuronal/bandit) y comparación por CV (IMPLEMENTADO)
│   ├── cost_utils.py      # Matrices de coste, coste esperado, optimización de umbral (IMPLEMENTADO)
│   └── xai_utils.py       # Helpers XAI (permutación, PDP 1D/2D, LIME, subgrupos por edad, inverse-transform) — usados por 08 (IMPLEMENTADO)
├── results/               # Resultados generados
│   ├── figures/           # Gráficas (incluye prep_*.png del preprocesado)
│   ├── tables/            # Tablas de resultados (incluye prep_*.csv del preprocesado)
│   ├── models/            # Artefactos serializados: preprocessing_pipeline.joblib, modelo_coste1.joblib, modelo_coste10.joblib
│   └── predicciones/      # cs_produccion1.csv y cs_produccion2.csv (entregables)
└── report/                # Informe de entrega
    ├── informe_visual.html            # Informe visual autocontenido (HTML)
    ├── informe_visual.pdf             # El mismo informe visual en PDF
    ├── Informe_Tecnico_XAI_Credito.tex # Informe técnico-académico en LaTeX
    └── figuras/                       # Figuras del informe técnico
```

### Convención de notebooks

- `01`–`07`: notebooks de trabajo, exploración y desarrollo incremental, uno por bloque temático:
  `01` EDA, `02` preprocesado, `03` modelo coste 1, `04` modelo coste 10, `05` auditoría subrogado,
  `06` contrafactuales, `07` SHAP.
- `08_otras_tecnicas.ipynb`: notebook adicional (no exigido, invitado por el enunciado como "otras
  técnicas que los estudiantes consideren oportunas") con importancia por permutación, PDP/ICE +
  interacción 2D, LIME (contraste crítico frente a SHAP) y subgrupos por tramo de edad. Autocontenido
  e independiente de `06`/`07`.
- `99_ENTREGA.ipynb`: notebook único consolidado que se entrega, **ya completo y ejecutado sin
  errores** (123 celdas, 13 secciones). Cada sección incorpora el **código real ejecutable**
  (extraído de `src/` y de los notebooks `01`–`08`) que produce sus resultados, además de las
  figuras, las tablas y la narrativa. Es autocontenido: puede ejecutarse y leerse **sin el resto
  del repositorio**.

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
- **Notebooks**: los **ocho notebooks de trabajo `01`–`08` están completos y ejecutados de principio
  a fin sin errores**. El `01` tiene el análisis exploratorio (carga/validación, EDA básico, valores
  centinela y outliers, EDA avanzado con comparación train/producción, correlaciones, PCA lineal y no
  lineal, recomendaciones de nulos y normalización, conclusiones). El `02` construye el pipeline de
  preprocesado (`src/preprocessing.py`) y deja sus artefactos: `data/processed/train.parquet`,
  `test.parquet` y `produccion.parquet`, más `results/models/preprocessing_pipeline.joblib` (y tablas
  `prep_*.csv` / figuras `prep_*.png`). El `03` (coste simétrico) y el `04` (coste asimétrico) comparan
  por validación cruzada anti-fuga varias familias de modelos (logística, XGBoost, dos MLP Keras y un
  bandit LinUCB propio) y eligen el umbral por coste esperado; XGBoost gana en ambos escenarios. Generan
  los dos entregables `results/predicciones/cs_produccion1.csv` y `cs_produccion2.csv`, los modelos
  `modelo_coste1.joblib` / `modelo_coste10.joblib`, la tabla `results/tables/umbrales_coste.csv` y las
  tablas/figuras `mod_03_*` / `mod_04_*`. El `05` audita ambos modelos con un árbol subrogado (barrido
  de profundidad y peso de clase, reglas legibles traducidas a escala original, tabla de fidelidad
  consolidada) y encuentra que el modelo tiene "dos personalidades de riesgo" según el umbral. El `06`
  genera **contrafactuales con DiCE** (2 casos por clase real y escenario, validación de plausibilidad y
  explicación no técnica al cliente); artefactos `contrafactuals_ejemplos.csv` y
  `contrafactuals_*.png`. El `07` calcula **SHAP** con `TreeExplainer` (summary global + waterfall
  locales sobre los mismos casos del `06`); artefactos `shap_values_global.csv`, `shap_summary_*.png`,
  `shap_local_*.png`. El `08` (no exigido por el enunciado) añade importancia por permutación, PDP/ICE +
  interacción 2D, LIME (con medición de inestabilidad frente a SHAP) y subgrupos por tramo de edad; ver
  `src/xai_utils.py` para las funciones reutilizables. Las decisiones internas de estos notebooks
  (D-5.1, D-6.1, D-6.2, D-7.1, D-8.1, D-8.2) están registradas en `docs/DECISIONES.md`.
- **Entrega**: el **`99_ENTREGA.ipynb`** (notebook único de entrega) está **completo y ejecutado de
  principio a fin sin errores** (123 celdas, 13 secciones): portada con datos del grupo, EDA,
  preprocesado, ambos modelos, tabla comparativa, las tres auditorías (subrogado, contrafactuales,
  SHAP), otras técnicas, decisiones y reflexión final. La novedad importante es que **cada sección
  incorpora el código real ejecutable** (extraído de `src/` y de los notebooks `01`–`08`) que genera
  sus resultados, junto con las figuras, tablas y la narrativa. Es **autocontenido**: el profesor
  puede ejecutarlo y leerlo **sin necesidad del resto del repositorio**. El informe de entrega
  acompaña al notebook en tres formatos: `report/informe_visual.html`, `report/informe_visual.pdf`
  y el informe técnico en LaTeX `report/Informe_Tecnico_XAI_Credito.tex` (PDF compilado en la raíz,
  `Informe_Tecnico_XAI_Credito.pdf`). **No queda nada pendiente.**

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

Lo ÚNICO que NO se versiona es `docs/_fuentes/`: el material bruto privado del profesor
(transparencias, notebooks de ejemplo y los datasets originales). Quien lo necesite debe
conseguirlo aparte; los notebooks **no dependen de esa carpeta** para ejecutarse (leen de `data/`).
Todos los artefactos generados por los notebooks `01`–`08` y por el `99_ENTREGA.ipynb` quedan
versionados en `results/`, de modo que el repositorio es reproducible de principio a fin.
