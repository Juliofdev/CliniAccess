import cv2 as cv
import os
import numpy as np
from time import time

# -------------------------
# Rutas relativas
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataRuta = os.path.join(BASE_DIR, "Data")

# Lista de carpetas dentro de Data (cada carpeta = una persona)
listaData = os.listdir(dataRuta)

ids = []
rostrosData = []
id_persona = 0

tiempoInicial = time()

# -------------------------
# Lectura de rostros
# -------------------------

for carpeta in listaData:
    rutaCompleta = os.path.join(dataRuta, carpeta)
    print("\nIniciando lectura de:", carpeta)

    for archivo in os.listdir(rutaCompleta):
        print("Imagen:", carpeta + "/" + archivo)

        # Cargar imagen en escala de grises
        rutaImagen = os.path.join(rutaCompleta, archivo)
        imagen = cv.imread(rutaImagen, 0)

        if imagen is None:
            print("⚠ No se pudo leer:", rutaImagen)
            continue

        ids.append(id_persona)
        rostrosData.append(imagen)

    id_persona += 1

    tiempoFinalLectura = time()
    print("Tiempo total de lectura:", tiempoFinalLectura - tiempoInicial)


# -------------------------
# Entrenamiento con EigenFace
# -------------------------

print("\nIniciando el entrenamiento... espere")

entrenador = cv.face.EigenFaceRecognizer_create()
entrenador.train(rostrosData, np.array(ids))

tiempoFinalEntrenamiento = time()
print("Tiempo total entrenamiento:", tiempoFinalEntrenamiento - tiempoFinalLectura)


# -------------------------
# Guardar archivo de reconocimiento
# -------------------------

modeloRuta = os.path.join(BASE_DIR, "EntrenamientoEigenFaceRecognizer.xml")
entrenador.write(modeloRuta)

print("\nEntrenamiento concluido ✔")
print("Modelo guardado en:", modeloRuta)
