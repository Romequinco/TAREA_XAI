# Reglas de denegacion - escenario coste1 (simetrico)

Reglas extraidas del **arbol subrogado** (max_depth=5) que imita al modelo XGBoost de caja negra bajo el umbral de decision **0.4934** (la caja negra deniega al **2.38%** de los solicitantes de test).

Cada regla es una rama del arbol que termina en DENEGAR, con las condiciones de una misma variable consolidadas en un intervalo. El **soporte** es cuantos solicitantes de test caen en esa regla; la **pureza** es que fraccion de ellos la caja negra tambien deniega (fidelidad local de la regla).

> **Como se lee:** si un solicitante cumple TODAS las condiciones de una regla, el modelo le deniega el credito. Estas 8 reglas cubren 543 denegaciones del conjunto de test.

---

### Regla 1  (soporte: 118 solicitantes | pureza: 100.0%)

Se DENIEGA el credito si:

- N. impagos 90+ dias >= 2
- N. moras 60-89 dias >= 2
- Utilizacion de credito revolving > 0.91

### Regla 2  (soporte: 82 solicitantes | pureza: 77.6%)

Se DENIEGA el credito si:

- N. impagos 90+ dias = 1
- N. moras 60-89 dias >= 2
- N. moras 30-59 dias <= 1
- Utilizacion de credito revolving > 0.77

### Regla 3  (soporte: 73 solicitantes | pureza: 93.3%)

Se DENIEGA el credito si:

- N. impagos 90+ dias = 1
- N. moras 60-89 dias >= 2
- N. moras 30-59 dias >= 2

### Regla 4  (soporte: 59 solicitantes | pureza: 65.7%)

Se DENIEGA el credito si:

- N. impagos 90+ dias >= 2
- N. moras 60-89 dias <= 1
- N. moras 30-59 dias <= 1
- Edad <= 40.50

### Regla 5  (soporte: 58 solicitantes | pureza: 75.0%)

Se DENIEGA el credito si:

- N. impagos 90+ dias >= 2
- N. moras 60-89 dias <= 1
- N. moras 30-59 dias >= 2
- Edad > 43.50

### Regla 6  (soporte: 53 solicitantes | pureza: 98.0%)

Se DENIEGA el credito si:

- N. impagos 90+ dias >= 2
- N. moras 60-89 dias <= 1
- N. moras 30-59 dias >= 2
- Edad <= 43.50

### Regla 7  (soporte: 50 solicitantes | pureza: 75.0%)

Se DENIEGA el credito si:

- N. impagos 90+ dias <= 1
- N. moras 30-59 dias >= 4
- Utilizacion de credito revolving > 0.73
- Ratio de deuda > 0.58

### Regla 8  (soporte: 50 solicitantes | pureza: 96.4%)

Se DENIEGA el credito si:

- N. impagos 90+ dias >= 2
- N. moras 60-89 dias >= 2
- Utilizacion de credito revolving <= 0.91
