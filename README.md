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
│   ├── preprocessing.py   # Pipeline de preprocesado fit/transform
│   ├── modeling.py        # Entrenamiento y evaluación de modelos
│   ├── cost_utils.py      # Matrices de coste, coste esperado, optimización de umbral
│   └── xai_utils.py       # Wrappers de SHAP, modelo subrogado y contrafactuales
├── results/               # Resultados generados
│   ├── figures/           # Gráficas
│   ├── tables/            # Tablas de resultados
│   └── predicciones/      # cs_produccion1.csv y cs_produccion2.csv (entregables, sí versionados)
└── report/                # Material auxiliar para el informe/entrega final
```

### Convención de notebooks

- `01`–`07`: notebooks de trabajo, exploración y desarrollo incremental (uno por bloque temático:
  EDA, preprocesado, modelado, coste, subrogado, contrafactuales, SHAP, etc.).
- `99_ENTREGA.ipynb`: notebook único consolidado que se entrega, con el código final, la
  justificación de los desarrollos, las tablas de resultados y la reflexión final.

## Cómo ejecutar

_(Pendiente de completar cuando el pipeline esté implementado)._
