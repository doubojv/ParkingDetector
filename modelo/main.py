# ----------- A principio, vamos preparar e tratar nossos dados

import os
import pickle
import numpy as np
# Usaremos a Scikit-image em vez de open-cv para tratar as imagens  
from skimage.io import imread
from skimage.transform import resize

#Import para treino e testes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Salvaremos o diretorio na variavel input_dir 
input_dir = 'C:/Users/João Vitor Coutinho/Documents/Códigos/Visão/ParkingDetector/clf-data'

# Criamos um vetor para as categorias da vaga do estacionamento 
categorias = ['vazio', 'cheio']

# Vetores auxiliares 
dados = []
rotulos = []

# Aqui vamos iterar sobre cada arquivo(imagem) de cada categoria(vazio ou cheio) 
for categoria_index, categoria in enumerate(categorias):
    for arquivo in os.listdir(os.path.join(input_dir, categoria)):
        img_path = os.path.join(input_dir, categoria, arquivo)
        # criamos um diretorio particular pra cada imagem
        img = imread(img_path)
        img = resize(img, (15,15))
        # Acima nós redimensionamos para 15x15 e a convertemos usando imread do scikit-image 
        dados.append(img.flatten())
        rotulos.append(categoria_index)
        # Aqui vamos transformar cada imagem em um vetor unidimensional e joga-lo para o fim da lista do vetor auxiliar Dados
        # pra futura manipulação com Machine Learning

dados = np.asarray(dados) # Aqui a gente converte ambos vetores auxiliares para vetores do tipo numpy
rotulos = np.asarray(rotulos)


# ----------- Agora vamos aos treinos e testes e ao machine learning propriamente dito
x_treino, x_teste, y_treino, y_teste = train_test_split(dados, rotulos, test_size= 0.2, shuffle= True, stratify=rotulos)

# Iremos usar o classificador SVM 
classificador = SVC()

# Criação de dicionarios de parametros, onde irei criar 12 (3 gamma x 4 C) tipos de classificadores diferentes 
# Gamma e C são hiperparametros, ou seja, parametros que sao definidos previamente antes do treinamento de dados
parametros = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

# Aqui vai acontecer a pesquisa em grade pra achar qual melhor combinação de hiperparametros
grid_search = GridSearchCV(classificador, parametros)

grid_search.fit(x_treino, y_treino)

melhor_estimador = grid_search.best_estimator_

predict_y = melhor_estimador.predict(x_teste)

score = accuracy_score(predict_y, y_teste)  # accuracy de aproximadamente 99.95%

pickle.dump(melhor_estimador, open('./model.p', 'wb'))