# HMM-Aplicacion de Simulación- Mercados Financieros

**Descripción**  
Aplicación en Python que implementa un Hidden Markov Model (HMM) para simular regímenes de mercado, generar series de precios condicionadas a estados ocultos, realizar inferencias (Viterbi propio y adaptador a `hmmlearn`) y visualizar resultados mediante una interfaz gráfica (Tkinter + Matplotlib). Incluye análisis de métricas financieras, histograma de retornos, grafo de transiciones y una gráfica comparativa entre proporción simulada y probabilidad estacionaria teórica.

---

## Características principales
- Editor interactivo de matrices **A** (transición), **B** (emisión) y distribución inicial **π**.
- Simulación paso a paso, automática y por lotes (Simular 50, Simular Muchas).
- Generación de precios condicionados por régimen (retornos muestreados por estado).
- Inferencia con Viterbi propio y adaptador a `hmmlearn` (con fallback).
- Visualizaciones: precio con fondo coloreado por régimen, barras de observaciones, histograma up/down con conclusión dinámica, heatmap de transiciones, evolución comparativa (proporción acumulada simulada vs prob. estacionaria).
- Exportación de resultados a Excel (.xlsx) (pendiente mejorar construccion).

---

### Requisitos previos
- Python 3.9+ (recomendado 3.10).
- pip/conda

### Dependencias (ejemplo)
- numpy
- pandas
- matplotlib
- openpyxl
- networkx
- seaborn (opcional para heatmap)
- hmmlearn (opcional, solo si se desea usar ese backend)

## Instalación
1. Clonar el repositorio o Descargar ZIP
2. Ejecutar main.py
### Pasos de instalación
1. Clonar el repositorio:
2. Crear y activar entorno virtual (recomendado):
    - Windows: `\activate`
    - Ejecutar la aplicación:
    - `python src/main.py`

---

## Flujo de la simulación (resumen)
1. **Configurar matrices**: editar la matriz de transición **A**, la matriz de emisión **B** y la distribución inicial **π** en el editor de la GUI.
2. **Iniciar simulación**:
    - `Start`: ejecución automática (visualización en tiempo real).
    - `Step`: avanzar un período.
    - `Simular 50`: ejecutar 50 períodos y mostrar resumen final.
    - `Simular Muchas`: ejecutar múltiples runs para análisis de convergencia.
3. **Visualizar resultados**:
    - Panel principal: precio, observaciones y grafo coloreado por frecuencia empírica.
    - Ventana resumen: histograma up/down + evolución comparativa (proporción acumulada simulada vs prob. estacionaria).
    - Ventana de resultados: grafo a la izquierda, evolución a la derecha, métricas y recomendaciones dinámicas.
4. **Inferencia**:
    - `Inferir (Viterbi)`: ejecutar Viterbi propio y comparar con estados reales.
    - `Inferir (hmmlearn)`: intentar inferir con `hmmlearn` (si está instalado), con fallback automático.
5. **Exportar (pendiente mejorar construccion)**: guardar métricas, secuencias, matrices y logs a Excel.

---

## Especificaciones del modelo (valores por defecto)
**Estados ocultos**
- `['Mercado Alcista', 'Mercado Lateral', 'Mercado Bajista']`

**Observaciones**
- `['Retorno Positivo', 'Retorno Neutral', 'Retorno Negativo']`

**Matriz de transición A (por defecto)**
- Alcista → [Alcista: 0.6, Lateral: 0.3, Bajista: 0.1]
- Lateral → [Alcista: 0.3, Lateral: 0.4, Bajista: 0.3]
- Bajista → [Alcista: 0.1, Lateral: 0.3, Bajista: 0.6]

**Matriz de emisión B (por defecto)**
- Alcista → [Positivo: 0.7, Neutral: 0.2, Negativo: 0.1]
- Lateral → [Positivo: 0.3, Neutral: 0.4, Negativo: 0.3]
- Bajista → [Positivo: 0.1, Neutral: 0.2, Negativo: 0.7]

**Distribución inicial π (por defecto)**
- `[Alcista: 0.4, Lateral: 0.3, Bajista: 0.3]`

**Justificación breve**
- El model oculto de markov es un modelo probalisticos de estados ocultos muy util cuando no podemos observar los estados o regimenes financieros.
- Las filas suman 1 y reflejan persistencia de regímenes (alta probabilidad en la diagonal). Las emisiones reflejan la intuición financiera: alcista produce mayor probabilidad de retornos positivos, bajista de retornos negativos, lateral mezcla neutrales. π refleja una ligera inclinación inicial hacia alcista.

---

## Organización del proyecto (estructura)
- `README.md` — documentación principal.
- `src/` — código fuente:
    - `main.py` — interfaz Tkinter y orquestador (`HMMApp`).
    - `model.py` — clase `HMMModel`: parámetros, simulador, generación de precios y análisis.
    - `inference.py` — Viterbi propio y adaptador a `hmmlearn`.
    - `viz.py` — funciones de visualización (Matplotlib).
    - `analysis_hmmlearn.py` — comparativas entre backends y utilidades.
---

## Buenas prácticas y recomendaciones
- Validar que las filas de **A** y **B** sumen 1 antes de aplicar (la app normaliza automáticamente, pero conviene revisar).
- Si se usa `hmmlearn`, instalar la versión compatible
---