import cv2 as cv
import os
import imutils

# -------------------------
# RUTAS RELATIVAS
# -------------------------

# Carpeta base donde está este archivo .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta Data dentro del proyecto
dataRuta = os.path.join(BASE_DIR, "Data")

# Carpeta donde se guardarán las fotos del modelo
modelo = "RostrosUsuario"
rutaCompleta = os.path.join(dataRuta, modelo)

# Crear carpetas si no existen
os.makedirs(rutaCompleta, exist_ok=True)

# -------------------------
# CARGAR CLASIFICADOR
# -------------------------

# Ruta al clasificador Haarcascade
clasificadorRuta = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

ruidos = cv.CascadeClassifier(clasificadorRuta)

# -------------------------
# INICIAR CÁMARA
# -------------------------

camara = cv.VideoCapture(0)
id = 0

while True:
    respuesta, captura = camara.read()
    if not respuesta:
        break

    captura = imutils.resize(captura, width=640)
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCaptura = captura.copy()

    caras = ruidos.detectMultiScale(grises, 1.3, 5)

    for (x, y, w, h) in caras:
        cv.rectangle(captura, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = idCaptura[y:y + h, x:x + w]
        rostro = cv.resize(rostro, (160, 160), interpolation=cv.INTER_CUBIC)

        # Guardar foto
        cv.imwrite(os.path.join(rutaCompleta, f"imagen_{id}.jpg"), rostro)
        id += 1

    cv.imshow("Resultado rostro", captura)

    if id == 350:
        break

camara.release()
cv.destroyAllWindows()
