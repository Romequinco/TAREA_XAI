# Reglas de denegacion - escenario coste10 (asimetrico)

Reglas extraidas del **arbol subrogado** (max_depth=5) que imita al modelo XGBoost de caja negra bajo el umbral de decision **0.0906** (la caja negra deniega al **18.70%** de los solicitantes de test).

Cada regla es una rama del arbol que termina en DENEGAR, con las condiciones de una misma variable consolidadas en un intervalo. El **soporte** es cuantos solicitantes de test caen en esa regla; la **pureza** es que fraccion de ellos la caja negra tambien deniega (fidelidad local de la regla).

> **Como se lee:** si un solicitante cumple TODAS las condiciones de una regla, el modelo le deniega el credito. Estas 18 reglas cubren 3940 denegaciones del conjunto de test.

---

### Regla 1  (soporte: 1171 solicitantes | pureza: 64.7%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving > 0.85
- N. moras 30-59 dias <= 1
- N. impagos 90+ dias <= 1
- Edad <= 49.50

### Regla 2  (soporte: 975 solicitantes | pureza: 100.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving > 0.77
- N. moras 30-59 dias >= 2
- Edad <= 63.50
- Ingreso mensual <= 12506.50

### Regla 3  (soporte: 373 solicitantes | pureza: 100.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving > 0.69
- N. moras 30-59 dias <= 1
- N. impagos 90+ dias >= 2

### Regla 4  (soporte: 224 solicitantes | pureza: 51.8%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving <= 0.69
- N. impagos 90+ dias <= 1
- N. moras 30-59 dias <= 1
- N. moras 60-89 dias >= 2

### Regla 5  (soporte: 186 solicitantes | pureza: 100.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving: entre 0.28 y 0.69
- N. impagos 90+ dias >= 2

### Regla 6  (soporte: 157 solicitantes | pureza: 98.7%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving: entre 0.39 y 0.69
- N. impagos 90+ dias <= 1
- N. moras 30-59 dias >= 2

### Regla 7  (soporte: 134 solicitantes | pureza: 76.1%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving: entre 0.17 y 0.39
- N. impagos 90+ dias <= 1
- N. moras 30-59 dias >= 2

### Regla 8  (soporte: 98 solicitantes | pureza: 95.9%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving <= 0.17
- N. impagos 90+ dias >= 2
- Edad <= 53.50

### Regla 9  (soporte: 88 solicitantes | pureza: 98.9%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving: entre 0.69 y 0.77
- N. moras 30-59 dias >= 2
- Edad <= 63.50
- Ingreso mensual <= 6349.80

### Regla 10  (soporte: 83 solicitantes | pureza: 100.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving <= 0.69
- N. impagos 90+ dias <= 1
- N. moras 30-59 dias = 1
- N. moras 60-89 dias >= 2

### Regla 11  (soporte: 68 solicitantes | pureza: 72.1%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving <= 0.17
- N. impagos 90+ dias >= 2
- Edad > 53.50

### Regla 12  (soporte: 66 solicitantes | pureza: 74.2%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving <= 0.17
- N. impagos 90+ dias <= 1
- N. moras 30-59 dias >= 2
- Ingreso mensual <= 5562.78

### Regla 13  (soporte: 61 solicitantes | pureza: 83.6%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving: entre 0.69 y 0.77
- N. moras 30-59 dias >= 2
- Edad <= 63.50
- Ingreso mensual > 6349.80

### Regla 14  (soporte: 53 solicitantes | pureza: 100.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving > 0.69
- N. moras 30-59 dias <= 1
- N. impagos 90+ dias <= 1
- Edad > 49.50
- N. moras 60-89 dias >= 2

### Regla 15  (soporte: 52 solicitantes | pureza: 78.8%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving > 0.69
- N. moras 30-59 dias >= 2
- Edad > 63.50
- N. lineas de credito abiertas <= 9

### Regla 16  (soporte: 51 solicitantes | pureza: 98.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving > 0.69
- N. moras 30-59 dias >= 2
- Edad > 63.50
- N. lineas de credito abiertas >= 10

### Regla 17  (soporte: 50 solicitantes | pureza: 98.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving: entre 0.17 y 0.28
- N. impagos 90+ dias >= 2

### Regla 18  (soporte: 50 solicitantes | pureza: 96.0%)

Se DENIEGA el credito si:

- Utilizacion de credito revolving > 0.77
- N. moras 30-59 dias >= 2
- Edad <= 63.50
- Ingreso mensual > 12506.50
