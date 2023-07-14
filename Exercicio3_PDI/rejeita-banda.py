import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem
image = cv2.imread('teste.png', cv2.IMREAD_GRAYSCALE)
# arara_filtro.png
# Configurações do filtro rejeita-banda
cutoff_low = 30
cutoff_high = 70

# Realiza a transformada de Fourier na imagem
f = np.fft.fft2(image)
f_shift = np.fft.fftshift(f)

# Calcula as dimensões da imagem e o centro
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Cria uma máscara passa-baixa
mask_low = np.ones((rows, cols), np.uint8)
mask_low[crow - cutoff_low: crow + cutoff_low, ccol - cutoff_low: ccol + cutoff_low] = 0

# Cria uma máscara passa-alta
mask_high = np.zeros((rows, cols), np.uint8)
mask_high[crow - cutoff_high: crow + cutoff_high, ccol - cutoff_high: ccol + cutoff_high] = 1

# Combina as máscaras para obter o filtro rejeita-banda
mask_bandstop = mask_low * mask_high

# Aplica a máscara na transformada de Fourier
f_shift_filtered = f_shift * mask_bandstop

# Realiza a transformada inversa de Fourier
f_inverse = np.fft.ifftshift(f_shift_filtered)
image_filtered = np.fft.ifft2(f_inverse)
image_filtered = np.abs(image_filtered)

# Normaliza a imagem filtrada para exibir corretamente
image_filtered = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(image_filtered, cmap='gray')
plt.axis('off')
plt.title('Imagem Filtrada (Rejeita-Banda)')

plt.show()
