#import mahotas as mahotas
import numpy as np
import cv2
from matplotlib import pyplot as plt

#leitura imagem
def lerSalvarImagem():
    
    obj_img = cv2.imread("imgs/marina.jpg")

    #mostra informções imagem
    print('Largura em pixels: ', end='')
    print(obj_img.shape[1]) #largura da imagem
    print('Altura em pixels: ', end='')
    print(obj_img.shape[0]) #altura da imagem
    print('Quantidade de canais: ', end='')
    print(obj_img.shape[2])

    #mostra imagem
    cv2.imshow("Nome janela", obj_img)
    cv2.waitKey(0)  #mostra até pressionada uma tecla

    cv2.imwrite("imgs/marina2.jpg", obj_img) #salva imagem como...

#lerSalvarImagem()


def mascara(): # mostra imagem original e um circulo preto com imagem dentro
    img = cv2.imread('imgs/marina.jpg')
    cv2.imshow("Original", img)
    mascara = np.zeros(img.shape[:2], dtype="uint8")
    (cX, cY) = (img.shape[1] // 2, img.shape[0] // 2)
    cv2.circle(mascara, (cX, cY), 100, 255, -1)
    img_com_mascara = cv2.bitwise_and(img, img, mask=mascara)
    cv2.imshow("Máscara aplicada à imagem", img_com_mascara)
    cv2.waitKey(0)


mascara()

def escalaCinza(): #espaços de cores
    img = cv2.imread('imgs/marina.jpg')
    cv2.imshow("Original", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imshow("L*a*b*", lab)
    cv2.waitKey(0)

#escalaCinza()


def canaisImagemColorida():
    img = cv2.imread('imgs/marina.jpg')
    (canalAzul, canalVerde, canalVermelho) = cv2.split(img)
    cv2.imshow("Vermelho", canalVermelho)
    cv2.imshow("Verde", canalVerde)
    cv2.imshow("Azul", canalAzul)
    cv2.waitKey(0)

#canaisImagemColorida()

# gerar histograma na imagem cinza
def histograma():
    img = cv2.imread('imgs/marina.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converte P&B
    cv2.imshow("Imagem P&B", img) #Função calcHist para calcular o hisograma da imagem
    h = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histograma P&B")
    plt.xlabel("Intensidade")
    plt.ylabel("Qtde de Pixels")
    plt.plot(h)
    plt.xlim([0, 256])
    plt.show()
    cv2.waitKey(0)

#histograma()


def espelhaImagem():
    img = cv2.imread('imgs/marina.jpg')
    cv2.imshow("Original", img)
    flip_horizontal = img[::-1, :]  # comando equivalente abaixo
    #flip_horizontal = cv2.flip(img, 1)
    #cv2.imshow("Flip Horizontal", flip_horizontal)
    flip_vertical = img[:, ::-1]  # comando equivalente abaixo
    #flip_vertical = cv2.flip(img, 0)
    cv2.imshow("Flip Vertical", flip_vertical)
    flip_hv = img[::-1, ::-1]  # comando equivalente abaixo
    #flip_hv = cv2.flip(img, -1)
    #cv2.imshow("Flip Horizontal e Vertical", flip_hv)
    cv2.waitKey(0)

#espelhaImagem()


#Read the image and perform threshold
#img = cv2.imread('imgs/digital3.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blur = cv2.medianBlur(gray, 5) _,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#Search for contours and select the biggest one contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cnt = max(contours, key=cv2.contourArea)
# #Create a new mask for the result image h, w = img.shape[:2]
# mask = np.zeros((h, w), np.uint8)
# #Draw the contour on the new mask and perform the bitwise operation cv2.drawContours(mask, [cnt],-1, 255, -1)
# res = cv2.bitwise_and(img, img, mask=mask)
# #Display the result
# cv2.imwrite('IMD006.png', res)
# #cv2.imshow('img', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#suavizar imagem
def suavizarImagem():
    img = cv2.imread('imgs/digital3.jpg')
    img = img[::2,::2] # Diminui a imagem
    suave = np.vstack([ np.hstack([img, cv2.blur(img, ( 3, 3))]), np.hstack([cv2.blur(img, (5,5)), cv2.blur(img, ( 7, 7))]), np.hstack([cv2.blur(img, (9,9)), cv2.blur(img, (11, 11))]), ])
    cv2.imshow("Imagens suavisadas (Blur)", suave)
    cv2.waitKey(0)

#suavizarImagem()

def suavizarImagemGaus():
    img = cv2.imread('imgs/marina.jpg')
    img = img[::2,::2] # Diminui a imagem 2 imagem maior, 3 imagem menor
    suave = np.vstack([ np.hstack([img, cv2.GaussianBlur(img, ( 3, 3), 0)]), np.hstack([cv2.GaussianBlur(img, ( 5, 5), 0), cv2.GaussianBlur(img, ( 7, 7), 0)]), np.hstack([cv2.GaussianBlur(img, ( 9, 9), 0), cv2.GaussianBlur(img, (11, 11), 0)]), ])
    cv2.imshow("Imagem original e suavisadas pelo filtro Gaussiano", suave)
    cv2.waitKey(0)

#suavizarImagemGaus()


#suavização pel Mediana
def suavizarMediana():
    img = cv2.imread('imgs/marina.jpg')
    img = img[::3,::3] # Diminui a imagem
    suave = np.vstack([ np.hstack([img, cv2.medianBlur(img, 3)]), np.hstack([cv2.medianBlur(img, 5), cv2.medianBlur(img, 7)]), np.hstack([cv2.medianBlur(img, 9), cv2.medianBlur(img, 11)]), ])
    cv2.imshow("Imagem original e suavisadas pela mediana", suave)
    cv2.waitKey(0)

#suavizarMediana()


#suavizarBilateral mantem bordas preservadas
def suavizarBilateral():
    img = cv2.imread('imgs/digital3.jpg')
    img = img[::2,::2] # Diminui a imagem
    suave = np.vstack([ np.hstack([img, cv2.bilateralFilter(img, 3, 21, 21)]), np.hstack([cv2.bilateralFilter(img, 5, 35, 35), cv2.bilateralFilter(img, 7, 49, 49)]), np.hstack([cv2.bilateralFilter(img, 9, 63, 63), cv2.bilateralFilter(img, 11, 77, 77)]) ])
    cv2.imshow("Imagem original e suavisadas pela bilateral", suave)
    cv2.waitKey(0)

#suavizarBilateral()

#binarização com limiar, imagem, imagem suavizada, imagem binarizada e imagem binarizada invertida
def binarizaçãoLimiar():
    img = cv2.imread('imgs/digital3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
    (T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
    (T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)
    resultado = np.vstack([ np.hstack([suave, bin]), np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)]) ])
    cv2.imshow("Binarização Limiar da imagem", resultado)
    cv2.waitKey(0)

#binarizaçãoLimiar()

#Threshold adaptativo, temos: a imagem, a imagem suavizada, a imagem binarizada pela média e a imagem binarizada com Gauss
def binarizarAdaptativo():
    img = cv2.imread('imgs/digital3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
    suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
    bin1 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    bin2 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    resultado = np.vstack([ np.hstack([img, suave]), np.hstack([bin1, bin2]) ])
    cv2.imshow("Binarização adaptativa da imagem", resultado)
    cv2.waitKey(0)

#binarizarAdaptativo()

#Binarização método Otsu e Riddler-Calvard
def binarizarOtsuRid():
    img = cv2.imread('imgs/digital3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
    suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
    T = mahotas.thresholding.otsu(suave)
    temp = img.copy()
    temp[temp > T] = 255
    temp[temp < 255] = 0
    temp = cv2.bitwise_not(temp)
    T = mahotas.thresholding.rc(suave)
    temp2 = img.copy()
    temp2[temp2 > T] = 255
    temp2[temp2 < 255] = 0
    temp2 = cv2.bitwise_not(temp2)
    resultado = np.vstack([ np.hstack([img, suave]), np.hstack([temp, temp2]) ])
    cv2.imshow("Binarização com método Otsu e Riddler-Calvard", resultado)
    cv2.waitKey(0)

#binarizarOtsuRid()



#segmentação detecção de bordas


#sobel
def sobel():
    img = cv2.imread('imgs/digital3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX, sobelY)
    resultado = np.vstack([ np.hstack([img, sobelX]), np.hstack([sobelY, sobel]) ])
    cv2.imshow("Sobel", resultado)
    cv2.waitKey(0)

#sobel()

#filtro laplaciano
def filtroLaplaciano():
    img = cv2.imread('imgs/digital3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    resultado = np.vstack([img, lap])
    cv2.imshow("Filtro Laplaciano", resultado)
    cv2.waitKey(0)

#canny
def detectorBordasCanny():
    img = cv2.imread('imgs/digital3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(img, (7, 7), 0)
    canny1 = cv2.Canny(suave, 20, 120)
    canny2 = cv2.Canny(suave, 70, 200)
    resultado = np.vstack([np.hstack([img, suave]), np.hstack([canny1, canny2])])
    cv2.imshow("Detector de Bordas Canny", resultado)
    cv2.waitKey(0)
#detectorBordasCanny()

###
#Função para facilitar a escrita nas imagem
def escreve(img, texto, cor=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0,cv2.LINE_AA)

#varias funçoes manipulação
  
    imgColorida = cv2.imread('imgs/digital3.jpg') #Carregamento da imagem
    #Se necessário o redimensioamento da imagem pode vir aqui.
    #Passo 1: Conversão para tons de cinza
    img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)
    #Passo 2: Blur/Suavização da imagem
    suave = cv2.blur(img, (7, 7))
    #Passo 3: Binarização resultando em pixels brancos e pretos
    T = mahotas.thresholding.otsu(suave)
    bin = suave.copy()
    bin[bin > T] = 255
    bin[bin < 255] = 0
    bin = cv2.bitwise_not(bin)
    #Passo 4: Detecção de bordas com Canny
    bordas = cv2.Canny(bin, 70, 150)
    #Passo 5: Identificação e contagem dos contornos da imagem
    #cv2.RETR_EXTERNAL = conta apenas os contornos externos
    (objetos, lx) = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #A variável lx (lixo) recebe dados que não são utilizados
    escreve(img, "Imagem em tons de cinza", 0)
    escreve(suave, "Suavizacao com Blur", 0)
    escreve(bin, "Binarizacao com Metodo Otsu", 255)
    escreve(bordas, "Detector de bordas Canny", 255)
    temp = np.vstack([ np.hstack([img, suave]), np.hstack([bin, bordas]) ])
    cv2.imshow("Quantidade de objetos: "+str(len(objetos)), temp)
    cv2.waitKey(0)
    imgC2 = imgColorida.copy()
    cv2.imshow("Imagem Original", imgColorida)
    cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
    escreve(imgC2, str(len(objetos))+" objetos encontrados!")
    cv2.imshow("Resultado", imgC2)
    cv2.waitKey(0)


escreve()

