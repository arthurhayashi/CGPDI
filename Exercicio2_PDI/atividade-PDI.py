import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('img_aluno1.jpg')
img2 = cv2.imread("img_aluno2.jpg")

def smooth_with_neighborhood_avg(image, neighborhood_size):
    # Aplicando zero-padding na imagem
    padded_image = cv2.copyMakeBorder(image, neighborhood_size//2, neighborhood_size//2, neighborhood_size//2, neighborhood_size//2, cv2.BORDER_REFLECT)

    # Criando uma matriz para armazenar a imagem suavizada
    smoothed_image = np.zeros_like(image)

    # Calculando a média da vizinhança para cada pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+neighborhood_size, j:j+neighborhood_size]
            smoothed_value = np.mean(neighborhood)
            smoothed_image[i, j] = smoothed_value

    return smoothed_image

def smooth_with_k_nearest_neighbors(data, k):
    smoothed_data = []
    n = len(data)

    for i in range(n):
        if k >= n:
            k_nearest = data  # Utiliza todos os elementos como vizinhos mais próximos
        else:
            left = max(0, i - k // 2)
            right = min(n, i + k // 2 + 1)
            k_nearest = data[left:right]  # Seleciona os k vizinhos mais próximos

        smoothed_value = np.mean(k_nearest, axis=0).astype(np.uint8)  # Calcula a média dos vizinhos
        smoothed_data.append(smoothed_value)

    return smoothed_data

def convolve2D(image, kernel):
    (image_height, image_width) = image.shape
    (kernel_height, kernel_width) = kernel.shape

    # Aplicando zero-padding na imagem
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Criando a matriz resultante
    result = np.zeros((image_height, image_width))

    # Realizando a convolução
    for i in range(image_height):
        for j in range(image_width):
            roi = padded_image[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(roi * kernel)

    return result

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)

    return (100 * ((image - min_val) / (max_val - min_val))).astype(np.uint8)


smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

robertsY = np.array((
	[0, 1],
    [-1, 0]), dtype="int")
robertsX = np.array((
	[1, 0],
    [0, -1]), dtype="int")
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

sobelY = np.array((
	[1, 2, 1],
	[0, 0, 0],
	[-1, -2, -1]), dtype="int")
prewittX = np.array((
	[-1, 0, 1],
	[-1, 0, 1],
	[-1, 0, 1]), dtype="int")
prewittY = np.array((
	[-1, -1, -1],
	[0, 0, 0],
	[1, 1, 1]), dtype="int")
aperiodica = np.array((
	[1, 1, 1],
	[0, 0, 0],
	[1, 1, 1]), dtype="int")


##Suavização média

img1_pb = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_pb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
avg_img1 = smooth_with_neighborhood_avg(img1_pb ,7)
avg_img2 = smooth_with_neighborhood_avg(img2_pb,7)

plt.figure(figsize=(10,10))

plt.subplot(221)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(222)
plt.imshow(avg_img1, cmap="gray")
plt.axis('off')
plt.title("Suavização média")

plt.subplot(223)
plt.imshow(cv2.cvtColor(img2_pb,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(224)
plt.imshow(avg_img2, cmap="gray")
plt.axis('off')
plt.title("Suavização média")

##Vizinhos proximos

smoothed_img1 = smooth_with_k_nearest_neighbors(img1_pb, 7)
smoothed_img2 = smooth_with_k_nearest_neighbors(img2_pb, 7)

plt.figure(figsize=(10,10))

plt.subplot(221)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(222)
plt.imshow(cv2.cvtColor(np.array(smoothed_img1), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Vizinhos próximos")

plt.subplot(223)
plt.imshow(cv2.cvtColor(img2_pb,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(224)
plt.imshow(cv2.cvtColor(np.array(smoothed_img2), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Vizinhos próximos")

##laplace

laplace1c = convolve2D(img1_pb, laplacian)
laplace2c = convolve2D(img2_pb, laplacian)
laplace1f = cv2.filter2D(img1_pb,-1, laplacian)
laplace2f = cv2.filter2D(img2_pb,-1, laplacian)

plt.figure(figsize=(26,10))

plt.subplot(231)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(232)
plt.imshow(laplace1c, cmap='gray')
plt.axis('off')
plt.title("Laplace (Convolved2d)")

plt.subplot(233)
plt.imshow(laplace1f, cmap='gray')
plt.axis('off')
plt.title("Laplace (cv2filter)")

plt.subplot(234)
plt.imshow(cv2.cvtColor(img2_pb,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(235)
plt.imshow(laplace2c, cmap='gray')
plt.axis('off')
plt.title("Laplace (Convolved2d)")

plt.subplot(236)
plt.imshow(laplace2f, cmap='gray')
plt.axis('off')
plt.title("Laplace (cv2filter)")

##roberts

roberts1Xc = convolve2D(img1_pb, robertsX)
roberts1Yc = convolve2D(img1_pb, robertsY)
roberts1XYc = cv2.addWeighted(roberts1Xc,0.5,roberts1Yc,0.5,0)
roberts1Xf = cv2.filter2D(img1_pb,-1 ,robertsX)
roberts1Yf = cv2.filter2D(img1_pb,-1, robertsY)
roberts1XYf = cv2.addWeighted(roberts1Xf,0.5,roberts1Yf,0.5,0)

roberts2Xc = convolve2D(img2_pb, robertsX)
roberts2Yc = convolve2D(img2_pb, robertsY)
roberts2XYc = cv2.addWeighted(roberts2Xc,0.5,roberts2Yc,0.5,0)
roberts2Xf = cv2.filter2D(img2_pb,-1 ,robertsX)
roberts2Yf = cv2.filter2D(img2_pb,-1, robertsY)
roberts2XYf = cv2.addWeighted(roberts2Xf,0.5,roberts2Yf,0.5,0)

plt.figure(figsize=(26,10))

plt.subplot(231)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(232)
plt.imshow(roberts1Xc, cmap='gray')
plt.axis('off')
plt.title("Roberts (Convolved2d)")

plt.subplot(233)
plt.imshow(roberts1XYf, cmap='gray')
plt.axis('off')
plt.title("Roberts (cv2filter)")

plt.subplot(234)
plt.imshow(cv2.cvtColor(img2_pb,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(235)
plt.imshow(roberts2Xc, cmap='gray')
plt.axis('off')
plt.title("Roberts (Convolved2d)")

plt.subplot(236)
plt.imshow(roberts2XYf, cmap='gray')
plt.axis('off')
plt.title("Roberts (cv2filter)")

##Prewitt

prewitt1Xc = convolve2D(img1_pb, prewittX)
prewitt1Yc = convolve2D(img1_pb, prewittY)
prewitt1XYc = cv2.addWeighted(prewitt1Xc,0.5,prewitt1Yc,0.5,0)
prewitt1Xf = cv2.filter2D(img1_pb,-1 ,prewittX)
prewitt1Yf = cv2.filter2D(img1_pb,-1, prewittY)
prewitt1XYf = cv2.addWeighted(prewitt1Xf,0.5,prewitt1Yf,0.5,0)

prewitt2Xc = convolve2D(img2_pb, prewittX)
prewitt2Yc = convolve2D(img2_pb, prewittY)
prewitt2XYc = cv2.addWeighted(prewitt2Xc,0.5,prewitt2Yc,0.5,0)
prewitt2Xf = cv2.filter2D(img2_pb,-1 ,prewittX)
prewitt2Yf = cv2.filter2D(img2_pb,-1, prewittY)
prewitt2XYf = cv2.addWeighted(prewitt2Xf,0.5,prewitt2Yf,0.5,0)

plt.figure(figsize=(26,10))

plt.subplot(231)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(232)
plt.imshow(prewitt1XYc, cmap='gray')
plt.axis('off')
plt.title("Prewitt (Convolved2d)")

plt.subplot(233)
plt.imshow(prewitt1XYf, cmap='gray')
plt.axis('off')
plt.title("Prewitt (cv2filter)")

plt.subplot(234)
plt.imshow(cv2.cvtColor(img2_pb,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(235)
plt.imshow(prewitt2XYc, cmap='gray')
plt.axis('off')
plt.title("Prewitt (Convolved2d)")

plt.subplot(236)
plt.imshow(prewitt2XYf, cmap='gray')
plt.axis('off')
plt.title("Prewitt (cv2filter)")

##sobel

sobel1Xc = convolve2D(img1_pb, sobelX)
sobel1Yc = convolve2D(img1_pb, sobelY)
sobel1XYc = cv2.addWeighted(sobel1Xc,0.5,sobel1Yc,0.5,0)
sobel1Xf = cv2.filter2D(img1_pb,-1 ,sobelX)
sobel1Yf = cv2.filter2D(img1_pb,-1, sobelY)
sobel1XYf = cv2.addWeighted(sobel1Xf,0.5,sobel1Yf,0.5,0)

sobel2Xc = convolve2D(img2_pb, sobelX)
sobel2Yc = convolve2D(img2_pb, sobelY)
sobel2XYc = cv2.addWeighted(sobel2Xc,0.5,sobel2Yc,0.5,0)
sobel2Xf = cv2.filter2D(img2_pb,-1, sobelX)
sobel2Yf = cv2.filter2D(img2_pb,-1,sobelY)
sobel2XYf = cv2.addWeighted(sobel2Xc,0.5,sobel2Yc,0.5,0)

plt.figure(figsize=(26,10))

plt.subplot(231)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(232)
plt.imshow(sobel1XYc, cmap='gray')
plt.axis('off')
plt.title("Sobel (Convolved2d)")

plt.subplot(233)
plt.imshow(sobel1XYf, cmap='gray')
plt.axis('off')
plt.title("Sobel (cv2filter)")

plt.subplot(234)
plt.imshow(cv2.cvtColor(img2_pb,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(235)
plt.imshow(sobel2XYc, cmap='gray')
plt.axis('off')
plt.title("Sobel (Convolved2d)")

plt.subplot(236)
plt.imshow(sobel2XYf, cmap='gray')
plt.axis('off')
plt.title("Sobel (cv2filter)")

plt.show()