import cv2
import numpy as np
import sys

# Função para aplicar a correção gamma na imagem
def gamma_correction(image, gamma=1.0):
    # Normaliza os valores dos pixels para o intervalo [0, 1]
    image_n = np.zeros((image.shape))
    image_n = cv2.normalize(image, image_n, 1, 0, cv2.NORM_MINMAX)
    
    # Aplica a correção gamma
    corrected_image = np.power(image_n, gamma)
    
    # Retorna a imagem com os valores normalizados novamente para [0, 255]
    return (corrected_image * 255).astype(np.uint8)

# Função para aplicar a regra 1
def regra1(image):
    # Crie uma máscara com base na comparação entre os canais Y e Cb
    mask = image[:, :, 0] > image[:, :, 2]
    
    # Aplique a máscara para definir os pixels
    image[mask] = 255
    image[~mask] = 0
    
    return image

# Função para aplicar a regra 2
def regra2(image):
    # Crie uma máscara com base na comparação entre os canais Cr e Cb
    mask = image[:, :, 1] > image[:, :, 2]
    
    # Aplique a máscara para definir os pixels
    image[mask] = 255
    image[~mask] = 0
    
    return image

# Função para aplicar a regra 3
def regra3(image):
    # Calcule as médias dos canais Y, Cr e Cb
    mean_y = np.mean(image[:, :, 0])
    mean_cr = np.mean(image[:, :, 1])
    mean_cb = np.mean(image[:, :, 2])
    
    # Crie uma máscara com base nas médias dos canais
    mask = (image[:, :, 0] > mean_y) & (image[:, :, 1] > mean_cr) & (image[:, :, 2] < mean_cb)
    
    # Aplique a máscara para definir os pixels
    image[mask] = 255
    image[~mask] = 0
    
    return image

# Função para aplicar a regra 4
def regra4(image):
    # Crie uma máscara com base na diferença entre os canais Cr e Cb
    mask = np.abs(image[:, :, 1] - image[:, :, 2]) > 80
    
    # Aplique a máscara para definir os pixels
    image[mask] = 255
    image[~mask] = 0
    
    return image

# Função para aplicar a regra 5
def regra5(image):
    # Crie uma máscara com base na comparação entre o canal V e um valor de limiar (Tsv)
    # Aqui, usamos o valor máximo do canal V como Tsv
    Tsv = np.max(image[:, :, 2])
    mask = image[:, :, 2] > Tsv
    
    # Aplique a máscara para definir os pixels
    image[mask] = 255
    image[~mask] = 0
    
    return image

# Carrega o arquivo de vídeo
video_path = 'fBackYardFire.avi'
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
output_path = 'resultado_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Define o valor de gamma maior que 1 para aumentar a intensidade do background
gamma_value = 0.1

while True:
    # Lê o próximo quadro do vídeo
    ret, frame = cap.read()

    # Verifica se o quadro foi lido corretamente
    if not ret:
        break

    # Aplica a correção gamma no quadro
    result_image = gamma_correction(frame, gamma=gamma_value)

    # Converte a imagem para o espaço de cores YCrCb
    ycrcb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2YCrCb)

    # Aplica as regras
    ycrcb_image = regra1(ycrcb_image)
    ycrcb_image = regra2(ycrcb_image)
    ycrcb_image = regra3(ycrcb_image)
    ycrcb_image = regra4(ycrcb_image)
    ycrcb_image = cv2.cvtColor(result_image, cv2.COLOR_YCrCb2BGR)
    ycrcb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2YUV)
    ycrcb_image = regra5(ycrcb_image)
    # result_image = cv2.cvtColor(result_image, cv2.COLOR_YUV2BGR)
    # Converte a imagem de volta para o espaço de cores BGR


    # Escreve o quadro resultante no novo arquivo de vídeo
    out.write(result_image.astype(np.uint8))

    # Exibe o quadro resultante
    cv2.imshow('Resultado', result_image)

    # Verifique se o usuário pressionou a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os objetos VideoCapture e VideoWriter e fecha todas as janelas abertas
cap.release()
out.release()
cv2.destroyAllWindows()