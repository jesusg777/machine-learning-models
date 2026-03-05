# Modelos de Machine Learning y Deep Learning

Este repositorio contiene una colección integral de algoritmos de Inteligencia Artificial desarrollados en Python. El proyecto documenta la evolución desde los fundamentos de las redes neuronales (Perceptrón, ADALINE, funciones de activación) hasta la implementación de arquitecturas profundas como Perceptrones Multicapa (MLP) y Redes Neuronales Recurrentes (LSTM) aplicadas a problemas de clasificación, regresión y procesamiento de lenguaje natural (NLP).

## 📂 Estructura del Proyecto

Los algoritmos y modelos están organizados conceptualmente:

```text
.
├── ADALINE/
│   ├── adaline_filtro_adaptativo.py
│   ├── adaline_multiple_output.py
│   └── RedADALINE.py
├── Deep_Learning_LSTM/
│   ├── Daily_Climate_LSTM_Regression_JesusGelve...
│   └── DBPedia_LSTM_Classification_JesusGelve...
├── Funciones_Activacion/
│   ├── FuncionEscalon.py
│   ├── FuncionSigmoide.py
│   ├── ReLU.py
│   └── TangenteHiperbolica.py
├── Informes_Tecnicos/
│   └── Informe_Clasificacion_y_Regresion_MLP.pdf
├── Optimizacion/
│   ├── optimizacion.py
│   ├── optimizacion_.py
│   └── Superficie3D.py
├── Perceptron/
│   ├── Neurona_AND_3D.py
│   ├── NEURONA_AND_Y_OR.py
│   ├── Perceptron.py
│   ├── PerceptronTarea.py
│   ├── perceptron_dos_salidas.py
│   └── Tabla_Representacion_Binaria.py
├── Proyectos_MLP_Regresion/
│   ├── regresion_garment_employees_productiv...
│   ├── SolutionXOR.py
│   ├── TareaClasificacion.ipynb
│   └── XoR_MLP.ipynb
├── .gitignore
├── organizar_archivos.py
└── README.md
```

## 📋 Tabla de Contenidos

- [Requisitos](#-requisitos)
- [Uso](#-uso)
- [Contenido Detallado](#-contenido-detallado)
  - [Modelos Avanzados (LSTM y MLP)](#modelos-avanzados-lstm-y-mlp)
  - [Fundamentos (Perceptrón y ADALINE)](#fundamentos-perceptrón-y-adaline)
  - [Matemáticas y Optimización](#matemáticas-y-optimización)
- [Documentación](#-documentación)
- [Autores](#️-autores)

## 📦 Requisitos

El proyecto utiliza **Python 3**. Dependiendo del script o notebook que desees ejecutar, necesitarás instalar las siguientes librerías:

```bash
pip install numpy matplotlib pandas scikit-learn tensorflow jupyter
```

_Nota: Para guardar las animaciones en los scripts de optimización, asegúrate de tener `pillow` instalado._

## 🚀 Uso

Los scripts independientes pueden ejecutarse directamente desde la terminal ubicándose en la **carpeta raíz** del proyecto. Para los modelos avanzados, se recomienda iniciar un servidor de Jupyter.

**Ejemplo de ejecución básica (ADALINE):**

```bash
python ADALINE/adaline_filtro_adaptativo.py
```

**Ejemplo para notebooks (MLP/LSTM):**

```bash
jupyter notebook Proyectos_MLP_Regresion/XoR_MLP.ipynb
```

## 📖 Contenido Detallado

### Modelos Avanzados (LSTM y MLP)

Implementaciones de Deep Learning para resolver problemas complejos no lineales y análisis de secuencias.

| Archivo/Carpeta                                | Descripción                                                                                                          |
| :--------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| `Deep_Learning_LSTM/Daily_Climate...`          | Red Recurrente LSTM diseñada para la predicción y regresión de series temporales climáticas diarias.                 |
| `Deep_Learning_LSTM/DBPedia_LSTM...`           | Modelo LSTM aplicado a tareas de Procesamiento de Lenguaje Natural (NLP) para la clasificación de textos en DBPedia. |
| `Proyectos_MLP_Regresion/XoR_MLP.ipynb`        | Solución del clásico problema no lineal XOR utilizando un Perceptrón Multicapa (MLP).                                |
| `Proyectos_MLP_Regresion/regresion_garment...` | Análisis predictivo de la productividad de empleados del sector textil mediante modelos de regresión.                |

### Fundamentos (Perceptrón y ADALINE)

Algoritmos clásicos de clasificación binaria, compuertas lógicas y procesamiento de señales.

| Archivo                                      | Descripción                                                                                                    |
| :------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `Perceptron/Perceptron.py`                   | Implementación base para resolver la compuerta lógica **AND** con visualización 2D de la frontera de decisión. |
| `Perceptron/NEURONA_AND_Y_OR.py`             | Modelado de compuertas lógicas simultáneas.                                                                    |
| `Perceptron/PerceptronTarea.py`              | Perceptrón en 3D entrenado con datos sintéticos generados en formas geométricas (Cubo, Esfera, Tetraedro).     |
| `Perceptron/Tabla_Representacion_Binaria.py` | Red neuronal para reconocer dígitos (0-9) en una matriz binaria de 5x3, evaluando robustez ante ruido.         |
| `ADALINE/adaline_filtro_adaptativo.py`       | ADALINE implementado como filtro adaptativo para la eliminación de ruido en señales sinusoidales.              |

### Matemáticas y Optimización

Scripts interactivos para comprender la mecánica del entrenamiento neuronal y el paso hacia adelante (forward pass).

| Archivo/Carpeta                 | Descripción                                                                                                                                      |
| :------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| `Funciones_Activacion/`         | Scripts individuales (`ReLU.py`, `FuncionSigmoide.py`, etc.) que grafican y definen las funciones de activación más utilizadas en la literatura. |
| `Optimizacion/optimizacion_.py` | Visualización 3D animada de optimización sobre una superficie compleja. Muestra trayectorias de convergencia (Genera GIF).                       |
| `Optimizacion/Superficie3D.py`  | Descenso por gradiente en una superficie 3D utilizando derivadas parciales específicas.                                                          |

## 📄 Documentación

En la carpeta `Informes_Tecnicos/` se encuentra el documento `Informe_Clasificacion_y_Regresion_MLP.pdf`, el cual detalla la metodología, el ajuste de hiperparámetros y el análisis de resultados obtenidos en los experimentos con Perceptrones Multicapa.

## ✒️ Autores

Ingeniería de Software e Inteligencia Artificial:

- **Juan David Yepez Velez**
- **Jesus David Gelves Cajiao**

---

_Este repositorio recopila implementaciones desarrolladas a lo largo del curso de Modelos de Machine Learning (Semestre 7)._
