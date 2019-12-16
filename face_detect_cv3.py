#IMPORTAMOS "opencv".
import cv2

#INTRODUCIMOS RUTA A LA IMAGEN Y EL ARCHIVO "xml".
imagePath = #<ruta a la imagen fuente>
cascPath = "haarcascade_frontalface_default.xml"

#CARGAMOS CLASIFICADOR.
faceCascade = cv2.CascadeClassifier(cascPath)

#LEEMOS IMAGEN
image = cv2.imread(imagePath)
#CONVERTIMOS IMAGEN A ESCALA DE GRISES
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#DETECTAMOS ROSTROS EN LA IMAGEN
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30))

#NÚMERO DE ROSTROS ENCONTRADOS
print("Found {0} faces!".format(len(faces)))

#MOSTRAMOS CONTENIDO DE "faces":
print("RECTANGLES:\n",faces)

#MARCAMOS LOS ROSTROS CON UN RECTANGULO
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#MOSTRAMOS RESULTADO.
cv2.imshow("Face deteccion", image)
cv2.waitKey(0)
