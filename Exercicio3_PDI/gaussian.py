import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem
image = cv2.imread('teste.png', cv2.IMREAD_GRAYSCALE)

# Configurações do filtro de ruído gaussiano
mean = 0
stddev = 20

# Gera o ruído gaussiano
noise = np.random.normal(mean, stddev, size=image.shape)

# Aplica o ruído à imagem
image_noisy = image + noise

# Normaliza a imagem para o intervalo 0-255
image_noisy = cv2.normalize(image_noisy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem com ruído
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(image_noisy, cmap='gray')
plt.axis('off')
plt.title('Imagem com Ruído Gaussiano')

plt.show()
