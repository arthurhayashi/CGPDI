import cv2
import numpy as np
import sys

def gamma_correction(image, gamma=1.0):
    # Normaliza os valores dos pixels para o intervalo [0, 1]
    image_n = np.zeros((image.shape))
    image_n = cv2.normalize(image, image_n, 1, 0, cv2.NORM_MINMAX)
    
    # Aplica a correção gamma
    corrected_image = np.power(image_n, gamma)
    
    # Retorna a imagem com os valores normalizados novamente para [0, 255]
    return (corrected_image * 255).astype(np.uint8)

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
    resultado = image
    return resultado

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
    resultado = image
    return resultado

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

def regra4(image):
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            cr_value = image[y, x, 1]  # cr
            cb_value = image[y, x, 2]  # cb
            if abs(np.int16(cb_value) - cr_value) > 80:
                image[y, x, 0] = 1
            else:
                image[y, x, 0] = 0
    resultado = image
    return resultado

def regra5(image):
    height, width, _ = image.shape
    valor_maximo = np.max(image[:, :, 2])  # Valor máximo do canal V (índice 2)
    tsv = np.mean(image[:, :, 2]) + (valor_maximo / 1.8)

    for y in range(height):
        for x in range(width):
            V_value = image[y, x, 2]
            if V_value > tsv:
                image[y, x, 0] = 1
            else:
                image[y, x, 0] = 0

    resultado = image
    return resultado

# Carrega o arquivo de vídeo
video_path = 'controlled3.avi'
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o arquivo de vídeo.")
    sys.exit()

# Obtém o tamanho do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define o codec e o objeto VideoWriter para gravar o vídeo resultante
output_path = 'resultado_regra1.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)
gamma_value = 0.1

while True:
    # Lê o próximo quadro do vídeo
    ret, frame = cap.read()

    # Verifica se o quadro foi lido corretamente
    if not ret:
        break
    result_image = gamma_correction(frame, gamma=gamma_value)
    ycrcb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Aplica a regra 1 no quadro
    result_frame = regra1(ycrcb_image)
    result_frame = regra2(ycrcb_image)
    result_frame = regra3(ycrcb_image)
    result_frame = regra4(ycrcb_image)
    ycrcb_image = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
    ycrcb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    result_frame = regra5(ycrcb_image)
    # result_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # Escreve o quadro resultante no novo arquivo de vídeo
    out.write(result_frame)

    # Exibe o quadro resultante
    cv2.imshow('Resultado', result_frame)

    # Verifique se o usuário pressionou a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os objetos VideoCapture e VideoWriter e fecha todas as janelas abertas
cap.release()
out.release()
cv2.destroyAllWindows()
