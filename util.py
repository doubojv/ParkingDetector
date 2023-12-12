import pickle

from skimage.transform import resize
import numpy as np
import cv2

# Definimos as constantes 
VAZIO = True
CHEIO = False

# Importamos nosso modelo baseado em 7000 imagens de vagas de estacionamentos treinado com SVC
MODEL = pickle.load(open("modelo/model.p", "rb"))

# Função Booleana para definir se a vaga está vazia ou cheia 
def empty_or_not(vaga_bgr):
# A principio recebemos a imagem de uma vaga do estacionamento em BGR

    dados_vaga = []  # Nesse trecho criamos o vetor que vai guardar os dados convertidos unidimensionalmente em numpy

    img_resized = resize(vaga_bgr, (15, 15, 3))
    dados_vaga.append(img_resized.flatten())
    dados_vaga = np.array(dados_vaga) # -------------------------------

    y_saida = MODEL.predict(dados_vaga) # Enviamos o dado para nosso modelo

    if y_saida == 0:
        return VAZIO
    else:
        return CHEIO

# Função que vai tratar e retornar as coordenadas de cada caixa delimitadora das vagas do estacionamento
def get_parking_spots_bboxes(componentes_conectados):
    # A função recebe uma tupla como argumento
    (totalLabels, label_ids, valores, centroid) = componentes_conectados
    
    slots = []
    coef = 1     # Coeficiente de escala
    for i in range(1, totalLabels):

        # Aqui extraimos os pontos de coordenada 
        x1 = int(valores[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(valores[i, cv2.CC_STAT_TOP] * coef)
        w = int(valores[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(valores[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
        # e adicionamos ao vetor slots

    return slots