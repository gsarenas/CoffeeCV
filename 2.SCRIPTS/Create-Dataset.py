'''
Esse algoritmo de pre-processamento de imagens foi utilizaod para gerar
o dataset referido no artigo "Aplicação da visão computacional e inteligência 
artificial na classificação da maturação de grãos de café" em
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# Lendo a imagem
image_path = "..\\1.DATASET\\IMMATURE\\immature_1.jpg"
image = cv2.imread(image_path)

# Convertendo a imagem para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Invertendo a imagem em escala de cinza
gray_inverted = cv2.bitwise_not(gray)

# Cria a imagem binaria 'thresholded'
_, binary = cv2.threshold(gray_inverted, 50, 255, cv2.THRESH_BINARY)

# Encontra os contornos da imagem binaria
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mascara vazia com fundo branco
mask = np.ones_like(image) * 255

# Desenha a regiao do grao de cafe na mascara em preto
cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

# Mesclando a mascara com a imagem original 
final_output = cv2.bitwise_or(image, mask)

# Encontrando o maior contorno
largest_contour = max(contours, key=cv2.contourArea)

# Define os limites do maior contorno (bounding box)
x, y, w, h = cv2.boundingRect(largest_contour)

# Recorta a imagem final usando os limites da bounding box
cropped_final_output = final_output[y:y+h, x:x+w]

# Calcula a media RGB
mean_rgb = np.mean(cropped_final_output, axis=(0, 1))
print("Media RGB:", mean_rgb)

# Especifica a classe para a imagem
image_class = "Mature"

# Carrega o arquivo CSV existente pra ler o numero da ultima amostra
last_sample_number = 0
try:
    with open("test_ML_mean_rgb_data.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Pula o cabecalho
        header = next(csv_reader, None)
        
        for row in csv_reader:
            last_sample_number = int(row[0])
except FileNotFoundError:
    pass

# Incrementa o numero da amostra para a imagem atual
current_sample_number = last_sample_number + 1

# Grava a media RGB no arquivo CSV
with open("mean_rgb_data.csv", "a", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Grava o cabecalho se estiver ausente
    if csv_file.tell() == 0:
        csv_writer.writerow(["Sample", "R", "G", "B", "Class"])
    
    csv_writer.writerow([current_sample_number, int(mean_rgb[2]), int(mean_rgb[1]), int(mean_rgb[0]), image_class])

# Mostra a imagem original, a binaria thresholded e a final (mesclada e recortada)
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(image[:, :, ::-1])
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(132)
plt.imshow(binary, cmap="gray")
plt.title("Imagem Binaria Limiarizada")
plt.axis("off")

plt.subplot(133)
plt.imshow(cropped_final_output[:, :, ::-1])
plt.title("Resultado Mascarado Recortado")
plt.axis("off")

plt.tight_layout()
plt.show()
