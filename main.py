import cv2 
from util import get_parking_spots_bboxes
from util import empty_or_not
video_path = 'videos/parking_crop_loop.mp4'   
miniMask = 'mini_mask.png'


miniMask = cv2.imread(miniMask, 0)  # Chamamos minimask a mascara de um fragmento pequeno do estacionamento 
cap = cv2.VideoCapture(video_path) 

componentes_conectados = cv2.connectedComponentsWithStats(miniMask, 4, cv2.CV_32S) 
"""Essa função acima vai retornar uma tupla com varias matrizes, passando a MiniMascara como a imagem que vai ser rotulada
    o 4 é o tipo de conectividade entre os pixels(vertical e horizontal), e o ultimo é o tipo de dado da matriz de rotulos
    (nesse caso 32 bits)"""


spots = get_parking_spots_bboxes(componentes_conectados)   # Aqui vamos ter uma matriz de coordenadas das vagas no estacionamento


flag = True
while flag: # Aqui fazemos o loop que vai abrir e fechar o video 
    flag, frame = cap.read()

    for spot in spots:  # Loop pra colocarmos uma caixa azul em cada vaga do estacionamento usando a matriz de rotulos 
        x1, y1, w, h = spot

        spot_cortado = frame[y1:y1 + h, x1:x1 + w, :]  # Separamos a vaga do estacionamento
        
        if empty_or_not(spot_cortado): # e Verificamos se está vazia ou não 
            frame = cv2.rectangle(frame, (x1,y1), (x1 + w , y1 + h), (0, 255, 0), 2) # Verde caso esteja
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x1 + w , y1 + h), (0, 0, 255), 2) # Vermelho caso contrario
    

    cv2.imshow('frame', frame)

    if(cv2.waitKey(25) & 0xFF == ord('c')): # Condição para fechar o video
        break


cap.release()
cv2.destroyAllWindows()
