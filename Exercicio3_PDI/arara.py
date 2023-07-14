import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega as imagens
imagem_original = cv2.imread('arara.png')
filtro = cv2.imread('arara_filtro.png', cv2.IMREAD_GRAYSCALE)

# Converte a imagem original para escala de cinza
imagem_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

# Calcula a transformada de Fourier da imagem original e do filtro
fft_imagem = np.fft.fft2(imagem_original)
fft_filtro = np.fft.fft2(filtro, s=imagem_original.shape)

# Aplica a filtragem no domínio da frequência
fft_resultado = fft_imagem * (1 - fft_filtro)

# Calcula a transformada inversa de Fourier para obter a imagem filtrada
imagem_filtrada = np.fft.ifft2(fft_resultado).real

# Normaliza os valores da imagem filtrada para o intervalo de 0 a 255
imagem_filtrada = cv2.normalize(imagem_filtrada, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(imagem_original, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(imagem_filtrada, cmap='gray')
plt.axis('off')
plt.title('Imagem Filtrada')

plt.show()
