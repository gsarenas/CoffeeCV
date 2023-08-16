import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Carregar o dataset do arquivo .csv
data = pd.read_csv("mean_rgb_data.csv")

# Obter os valores de R, G e B
red = data['R']
green = data['G']
blue = data['B']

# Normalizar os valores RGB [0, 255] para [0, 1]
norm_red = red / 255.0
norm_green = green / 255.0
norm_blue = blue / 255.0

# Tamanho do quadrado (em pixels)
square_size = 30

# Espaçamento entre os quadrados (em pixels)
spacing = 2

# Calcular o tamanho da imagem com base na quantidade de amostras
num_samples = len(data)
cols = int(np.ceil(np.sqrt(num_samples)))
rows = int(np.ceil(num_samples / cols))
image_width = cols * (square_size + spacing) - spacing
image_height = rows * (square_size + spacing) - spacing

# Criar uma imagem em branco com o tamanho adequado
image = Image.new('RGB', (image_width, image_height), color='white')
draw = ImageDraw.Draw(image)

# Fonte para a numeração
font = ImageFont.truetype("arial.ttf", 15)

# Desenhar cada quadrado com a cor correspondente, a numeração e a malha quadriculada
x_start, y_start = 0, 0
for i in range(num_samples):
    color = (int(norm_red[i] * 255), int(norm_green[i] * 255), int(norm_blue[i] * 255))
    
    # Desenhar quadrado colorido
    draw.rectangle([x_start, y_start, x_start + square_size, y_start + square_size], fill=color, outline='black')
    
    # Desenhar numeração dentro do quadrado
    draw.text((x_start + 5, y_start + 5), str(i + 1), font=font, fill='black')
    
    x_start += square_size + spacing
    if x_start + square_size > image_width:
        x_start = 0
        y_start += square_size + spacing

# Salvar a imagem
image.save('Color-Map-RGB.png')

# Mostrar a imagem
image.show()