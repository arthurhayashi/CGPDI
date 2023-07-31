import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def gamma_correction(image, gamma=1.0):
    # Normaliza os valores dos pixels para o intervalo [0, 1]
    image_n= np.zeros((image.shape))
    image_n=cv2.normalize(image,image_n, 1, 0,cv2.NORM_MINMAX)
    
    # Aplica a correção gamma
    corrected_image = np.power(image_n, gamma)
    # Retorna a imagem com os valores normalizados novamente para [0, 255]
    return (corrected_image * 255).astype(np.uint8)

# Carrega a imagem desejada
image_path = 'sol1.jpeg'
image = cv2.imread(image_path)
image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define o valor de gamma maior que 1 para aumentar a intensidade do background
gamma_value =  0.1

# Aplica a correção gamma na imagem
result_image = gamma_correction(image, gamma=gamma_value)
result_image_G= gamma_correction(image_g, gamma=gamma_value)

# Realiza a subtração da imagem original pela imagem com filtro
subtracted_image = cv2.subtract(image, result_image)
subtracted_image_G = cv2.subtract(image_g, result_image_G)

#image1 = cv2.cvtColor(subtracted_image, cv2.COLOR_BGR2GRAY)

diff_image_RGB = cv2.absdiff(image, subtracted_image)
diff_image2 = cv2.cvtColor(diff_image_RGB, cv2.COLOR_BGR2GRAY)
#diff_image_teste = np.zeros((diff_image_RGB.shape[0],diff_image_RGB.shape[1]))
#diff_image_teste = np.abs(0.114*diff_image_RGB[:,:,0] + 0.587*diff_image_RGB[:,:,1] + 0.299*diff_image_RGB[:,:,2])

diff_image_G = cv2.absdiff(image_g,subtracted_image_G)

diff_total= cv2.absdiff(diff_image_G, diff_image2)
diff_total=cv2.normalize(diff_total, None, 1, 0, cv2.NORM_MINMAX)
#_, binary_diff_total = cv2.threshold(diff_total, 145, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, binary_diff_total = cv2.threshold(diff_image2, 0, 25, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

imgYCC = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

height, width, _ = image.shape

# Definir a função regra1
def regra1(image):
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            y_value = image[y, x, 0]#y
            cb_value = image[y, x, 2]#cb
            if y_value > cb_value:
                image[y, x, 0] = 1
            else:
                image[y, x, 0] = 0
    resultado =image
    return(resultado)

#Definir a função regra2
def regra2(image):
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            cr_value = image[y, x, 1]#cr
            cb_value = image[y, x, 2]#cb
            if cr_value > cb_value:
                image[y, x, 0] = 1
            else:
                image[y, x, 0] = 0
    resultado =image
    return(resultado)

#Definir a função regra3
def regra3(image):
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            y_value = image[y, x, 0]#y
            cr_value = image[y, x, 1]#cr
            cb_value = image[y, x, 2]#cb
            media_y = np.mean(y_value)
            media_cr = np.mean(cr_value)
            media_cb = np.mean(cb_value)
            if y_value > media_y and cr_value>media_cr and cb_value<media_cb:
                image[y, x, 0] = 1
            else:
                image[y, x, 0] = 0
    resultado =image
    return(resultado)

#Definir a função regra4
def regra4(image):
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            cr_value = image[y, x, 1]#cr
            cb_value = image[y, x, 2]#cb
            #print(type(cr_value),type(cb_value))
            #print(cb_value, cr_value, np.abs(cb_value-cr_value))
            if abs(np.int16(cb_value) - cr_value)>80:
                image[y, x, 0] = 1
            else:
                image[y, x, 0] = 0
    resultado =image
    return(resultado)

#Definir a função regra5
def regra5(image):
    height, width, _ = image.shape
    valor_maximo=0
    for y in range(height):
        for x in range(width):
            V_value = image[y, x, 2]#v
            media_y = np.mean(V_value)
            if V_value > valor_maximo:
                valor_maximo = V_value
                tsv=media_y+(valor_maximo/1.8)
                if V_value>tsv:
                    image[y, x, 0] = 1
                else:
                    image[y, x, 0] = 0
    resultado =image
    return(resultado)

# Aplicando regras
image_copy = imgYCC.copy()
resultado1=regra1(image_copy)
resultado2=regra2(resultado1)
resultado3=regra3(resultado2)
resultado4=regra4(resultado3)



# Converter a imagem de volta para o espaço de cores BGR para exibir corretamente
resultado1 = cv2.cvtColor(resultado1, cv2.COLOR_YCrCb2BGR)
resultado1 = cv2.cvtColor(resultado1, cv2.COLOR_BGR2GRAY)

resultado2 = cv2.cvtColor(resultado2, cv2.COLOR_YCrCb2BGR)
resultado2 = cv2.cvtColor(resultado2, cv2.COLOR_BGR2GRAY)

resultado3 = cv2.cvtColor(resultado3, cv2.COLOR_YCrCb2BGR)
resultado3 = cv2.cvtColor(resultado3, cv2.COLOR_BGR2GRAY)

resultado4 = cv2.cvtColor(resultado4, cv2.COLOR_YCrCb2BGR)

resultado4 = cv2.cvtColor(resultado4, cv2.COLOR_BGR2YUV)

resultado5=regra5(resultado4)

resultado5 = cv2.cvtColor(resultado5, cv2.COLOR_YUV2BGR)
resultado5 = cv2.cvtColor(resultado5, cv2.COLOR_BGR2HSV)

aaa=cv2.inRange(resultado5,np.array([0,100,20]),np.array([25,255,255]))
bbb=cv2.bitwise_and(image,image,mask=aaa)
resultado5 = cv2.cvtColor(resultado5, cv2.COLOR_HSV2BGR)
cv2.imshow('aaa',bbb)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Mostra a imagem original, a imagem com filtro e a imagem resultante (subtraída)
cv2.imshow('Imagem Original', image)
cv2.imshow(f'Imagem com Gamma {gamma_value}', result_image)
cv2.imshow('Imagem Subtraída', subtracted_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem Subtraída', diff_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem Subtraída', diff_image_G)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem diff total', binary_diff_total)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem YCC', imgYCC)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem regra1', resultado1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem regra2', resultado2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem regra3', resultado3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem regra4', resultado4)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagem regra5', resultado5)
cv2.waitKey(0)
cv2.destroyAllWindows()