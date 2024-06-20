import cv2
from ultralytics import YOLO

modelPlates = YOLO('BestTrain/weights/plates/best.pt')
modelLN = YOLO('BestTrain/weights/letters_numbers/bestFull30.pt')

cap = cv2.VideoCapture(0)
ctexto = ""
# cap = cv2.VideoCapture('Videos/v3.mp4')

while cap.isOpened():

    success, frame = cap.read()
    if success:

        resultsPlates = modelPlates(frame, imgsz=640)

        # Accede a un elemento de la lista resultsPlates
        resultsPlates = resultsPlates[0]

        # Accede al atributo boxes del objeto Results
        boxes = resultsPlates.boxes

        for box in boxes:
            # Verifica si el objeto box tiene un atributo xyxy
            if box is not None:
                # Obtiene las coordenadas de la detección
                b = box.xyxy[0]

                # Detecta letras y números en la región de interés
                resultsLN = modelLN(b, imgsz=640)
                letras_numeros = resultsLN[0].plot()

                cv2.imshow("Letters and Numbers", letras_numeros)
            else:
                # El objeto box no es una detección válida
                x = y = w = h = 0
                
        # Dibuja las placas detectadas en la imagen
        placas = resultsPlates.plot()
        cv2.imshow("Plates", placas)

        tecla = cv2.waitKey(1)

        if tecla == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
