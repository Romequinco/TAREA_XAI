"""Pipeline de preprocesado (fit/transform) serializable.

Se ajusta una única vez sobre los datos de entrenamiento y se aplica de forma
idéntica tanto al conjunto de entrenamiento/test como a los datos de producción,
garantizando que ambos flujos reciben exactamente la misma transformación.
"""
