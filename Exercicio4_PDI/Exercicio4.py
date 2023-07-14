import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("circuito.tif")
img1_pb = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


#media 3x
plt.figure(figsize=(16,16))
plt.subplot(221)
plt.title('Imagem Original')
plt.imshow(img1_pb, cmap="gray", vmin=0, vmax=255)
plt.axis('off')

img1_blur1 = cv2.medianBlur(img1_pb, 3)
img1_blur2 =cv2.medianBlur(img1_blur1, 3)
img1_blur3 = cv2.medianBlur(img1_blur2, 3)

plt.subplot(222)
plt.title('Imagem Filtrada 1 vez')
plt.imshow(img1_blur1, cmap="gray", vmin=0, vmax=255)
plt.axis('off')

plt.subplot(223)
plt.title('Imagem Filtrada 2 vezes')
plt.imshow(img1_blur2, cmap="gray", vmin=0, vmax=255)
plt.axis('off')

plt.subplot(224)
plt.title('Imagem Filtrada 3 vezes')
plt.imshow(img1_blur3, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.show()


#filtro pontos isolados
img2 = cv2.imread("pontos.png")
img2_pb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(16,16))
plt.subplot(121)
plt.title('Original')
plt.imshow(img2_pb, cmap="gray", vmin=0, vmax=255)
plt.axis('off')

filtro = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

pontosFiltrado = cv2.filter2D(img2_pb, -1, filtro)

res, pontosLimiarizados = cv2.threshold(pontosFiltrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.subplot(122)
plt.title('Pontos Destacados')
plt.imshow(pontosLimiarizados, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.show()


#linhas verticais,horizontais
img2 = cv2.imread("linhas.png")
img2_pb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


filtro_horizontal = np.array([[-1, -1, -1],
                              [ 2,  2,  2],
                              [-1, -1, -1]], dtype=np.float32)

filtro_45_graus = np.array([[-1, -1,  2],
                            [-1,  2, -1],
                            [ 2, -1, -1]], dtype=np.float32)

filtro_vertical = np.array([[-1,  2, -1],
                            [-1,  2, -1],
                            [-1,  2, -1]], dtype=np.float32)

filtro_menos_45_graus = np.array([[ 2, -1, -1],
                                  [-1,  2, -1],
                                  [-1, -1,  2]], dtype=np.float32)


resultado_horizontal = cv2.filter2D(img2_pb, -1, filtro_horizontal)
resultado_45_graus = cv2.filter2D(img2_pb, -1, filtro_45_graus)
resultado_vertical = cv2.filter2D(img2_pb, -1, filtro_vertical)
resultado_menos_45_graus = cv2.filter2D(img2_pb, -1, filtro_menos_45_graus)


limiar = 127
_, img2_pb_limiarizada_horizontal = cv2.threshold(resultado_horizontal, limiar, 255, cv2.THRESH_BINARY)
_, img2_pb_limiarizada_45_graus = cv2.threshold(resultado_45_graus, limiar, 255, cv2.THRESH_BINARY)
_, img2_pb_limiarizada_vertical = cv2.threshold(resultado_vertical, limiar, 255, cv2.THRESH_BINARY)
_, img2_pb_limiarizada_menos_45_graus = cv2.threshold(resultado_menos_45_graus, limiar, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(16,16))
plt.subplot(221)
plt.title('Linhas Horizontais')
plt.imshow(img2_pb_limiarizada_horizontal, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(222)
plt.title('Linhas 45 Graus')
plt.imshow(img2_pb_limiarizada_45_graus, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(223)
plt.title('Linhas Verticais')
plt.imshow(img2_pb_limiarizada_vertical, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(224)
plt.title('Linhas -45 Graus')
plt.imshow(img2_pb_limiarizada_menos_45_graus, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.show()

#bordas de canny
img4 = cv2.imread("igreja.png")
img4_pb = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(16,16))
plt.subplot(121)
plt.title('Original')
plt.imshow(img4_pb, cmap="gray", vmin=0, vmax=255)
plt.axis('off')

limiar_min = 100
limiar_max = 200
igrejaBordasCanny = cv2.Canny(img4_pb, limiar_min, limiar_max)

plt.subplot(122)
plt.title('Bordas de Canny')
plt.imshow(igrejaBordasCanny, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.show()


#algoritmo de crescimento
img5 = cv2.imread('root.jpg')
img5_pb = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img5_pb,(5,5),0)
ret2,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

mask = np.zeros((img5.shape[0] + 2, img5.shape[1] + 2), np.uint8)

seed_point = (472, 216)
color = (0,0,0)
minColor = (0,0,0)
maxColor = (128,128,128)

plt.figure(figsize=(16,16)) 
plt.subplot(221)
plt.title('Imagem Original')
plt.imshow(img5_pb, cmap="gray")
plt.axis('off')

plt.subplot(222)
plt.title('Imagem com limiarização de Otsu')
plt.imshow(threshold, cmap="gray")
plt.axis('off')

cv2.floodFill(threshold, mask, seed_point, color, minColor, maxColor)

for y in range(img5.shape[0]):
    for x in range(img5.shape[1]):
        if mask[y, x].all():
            img5_pb[y, x] = 0


plt.subplot(223)
plt.title('Imagem com máscara do Crescimento de Região')
plt.imshow(mask, cmap="gray")
plt.axis('off')

plt.subplot(224)
plt.title('Imagem com Crescimento de Região')
plt.imshow(img5_pb, cmap="gray")
plt.axis('off')
plt.show()


#Método Otsu
img6 = cv2.imread('harewood.jpg')
img7 = cv2.imread('nuts.jpg')
img8 = cv2.imread('snow.jpg')
img9 = cv2.imread('lena.png')

img6_pb = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
img7_pb = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
img8_pb = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
img9_pb = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(16,16))
plt.subplot(221)
plt.title('harewood original')
plt.imshow(img6_pb, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(222)
plt.title('nuts original')
plt.imshow(img7_pb, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(223)
plt.title('snow original')
plt.imshow(img8_pb, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(224)
plt.title('img aluno original')
plt.imshow(img9_pb, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.show()

res1, img6_limiarizada = cv2.threshold(img6_pb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
res2, img7_limiarizada = cv2.threshold(img7_pb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
res3, img8_limiarizada = cv2.threshold(img8_pb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
res4, img9_limiarizada = cv2.threshold(img9_pb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(16,16))
plt.subplot(221)
plt.title('harewood limiarizada')
plt.imshow(img6_limiarizada, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(222)
plt.title('nuts limiarizada')
plt.imshow(img7_limiarizada, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(223)
plt.title('snow limiarizada')
plt.imshow(img8_limiarizada, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(224)
plt.title('img limiarizada')
plt.imshow(img9_limiarizada, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.show()