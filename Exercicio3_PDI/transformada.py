import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem em escala de cinza
img1 = cv2.imread('arara.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('barra1.png', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('barra2.png', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('barra3.png', cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread('barra4.png', cv2.IMREAD_GRAYSCALE)
img6 = cv2.imread('quadrado.png', cv2.IMREAD_GRAYSCALE)
img7 = cv2.imread('teste.png', cv2.IMREAD_GRAYSCALE)


# Aplica a transformada de Fourier
f1 = np.fft.fft2(img1)
f1_shift = np.fft.fftshift(f1)

f2 = np.fft.fft2(img2)
f2_shift = np.fft.fftshift(f2)

f3 = np.fft.fft2(img3)
f3_shift = np.fft.fftshift(f3)

f4 = np.fft.fft2(img4)
f4_shift = np.fft.fftshift(f4)

f5 = np.fft.fft2(img5)
f5_shift = np.fft.fftshift(f5)

f6 = np.fft.fft2(img6)
f6_shift = np.fft.fftshift(f6)

f7 = np.fft.fft2(img7)
f7_shift = np.fft.fftshift(f7)


# Calcula o espectro de magnitude
magnitude_spectrum1 = 20 * np.log(np.abs(f1_shift))

magnitude_spectrum2 = 20 * np.log(np.abs(f2_shift))

magnitude_spectrum3 = 20 * np.log(np.abs(f3_shift))

magnitude_spectrum4 = 20 * np.log(np.abs(f4_shift))

magnitude_spectrum5 = 20 * np.log(np.abs(f5_shift))

magnitude_spectrum6 = 20 * np.log(np.abs(f6_shift))

magnitude_spectrum7 = 20 * np.log(np.abs(f7_shift))

# Exibe a imagem original e o espectro de Fourier
plt.subplot(4,4,1)
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(4,4,3)
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(4,4,5)
plt.imshow(img3, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(4,4,7)
plt.imshow(img4, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(4,4,9)
plt.imshow(img5, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(4,4,11)
plt.imshow(img6, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(4,4,13)
plt.imshow(img7, cmap='gray')
plt.axis('off')
plt.title('Imagem Original')

plt.subplot(4, 4, 2)
plt.imshow(magnitude_spectrum1, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 4)
plt.imshow(magnitude_spectrum2, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 6)
plt.imshow(magnitude_spectrum3, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 8)
plt.imshow(magnitude_spectrum4, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 10)
plt.imshow(magnitude_spectrum5, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 12)
plt.imshow(magnitude_spectrum6, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 14)
plt.imshow(magnitude_spectrum7, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.show()
