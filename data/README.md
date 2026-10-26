# Datos

Los datos **no se incluyen en el repositorio**: son demasiado pesados (el CSV original ocupa ~110 MB) y el original contiene columnas con datos personales de clientes (nombre, dirección, etc.), que el script de limpieza elimina.

Para reproducir el análisis, coloca en esta carpeta:

| Fichero | Para qué sirve |
|---|---|
| `cadena_subministrament_2015_2018.csv` | Dataset original facilitado en la asignatura (≈180.000 pedidos, 58 variables) |
| `countries.geojson` | Polígonos de países (con campo `ADMIN`, estilo Natural Earth), para los mapas |
| `continents.geojson` | Polígonos de continentes, para el mapa continental |

`cadena_subministrament_2015_2018_diccionari.csv` (incluido) describe cada variable.

Después, ejecuta `python src/02_neteja.py` para generar `supply_chain_cleaned.csv`.
