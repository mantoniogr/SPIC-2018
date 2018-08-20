import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import math

def histogramaGris(img):
	H = np.zeros(500)

	height, width, channels =  img.shape

	for j in range(0, height):
		for i in  range(0, width):
			H[img[j,i,0]] = H[img[j,i,0]] + 1

	plt.suptitle('Histograma')

	plt.subplot(1, 2, 1)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.title("Original")
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.ylabel('Frecuencias')
	plt.xlabel('Niveles de intensidad')
	plt.axis([0, 500, 0, np.amax(H) + 100])
	plt.grid(True)
	plt.plot(H)

	plt.show()

	return

def histogramaColor(img):
	height, width, channels =  img.shape

	H_r = np.zeros(256)
	H_g = np.zeros(256)
	H_b = np.zeros(256)

	for j in range(0, height):
		for i in  range(0, width):
			H_b[img[j,i,0]] = H_b[img[j,i,0]] + 1
			H_g[img[j,i,1]] = H_g[img[j,i,1]] + 1
			H_r[img[j,i,2]] = H_r[img[j,i,2]] + 1

	plt.suptitle('Histograma')

	plt.subplot(1, 2, 1)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.title("Original")
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.ylabel('Frecuencias')
	plt.xlabel('Niveles de intensidad')

	max = 0
	if np.amax(H_r) > max:
		max = np.amax(H_r)
	elif np.amax(H_g) > max:
		max = np.amax(H_g)
	elif np.amax(H_b) > max:
		max = np.amax(H_b)

	plt.axis([0, 255, 0, max + 100])
	plt.grid(True)
	plt.plot(H_b)
	plt.plot(H_g)
	plt.plot(H_r)

	plt.show()

def gris(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height):
		for i in range(0, width):
			img_auxiliar[j,i,0] = 0.333*img[j, i, 0] + 0.333*img[j, i, 1] + 0.333*img[j, i, 2]

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def sumaR(img, img2):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height):
		for i in range(0, width):
			if(img2[j, i, 0] != 0):
				img_auxiliar[j, i, 2] = 255

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,1]
	img[:,:,2] = img_auxiliar[:,:,2]

	return img

def negativoGrises(img):
	height, width, channels =  img.shape
	negativo = np.copy(img)

	for j in range(0, height):
		for i in  range(0, width):
			negativo[j, i, 0] = 255 - img[j, i, 0]

	negativo[:,:,1] = negativo[:,:,0]
	negativo[:,:,2] = negativo[:,:,0]

	return negativo

def umbralGrises(img, umbral1, umbral2):
	height, width, channels =  img.shape
	imgAuxiliar = np.copy(img)

	for j in range(0, height):
		for i in  range(0, width):
			if img[j, i, 0] >= umbral1 and umbral2 >= img[j, i, 0]:
				imgAuxiliar[j, i, 0] = 255;
			else:
				imgAuxiliar[j, i, 0] = 0;

	imgAuxiliar[:,:,1] = imgAuxiliar[:,:,0]
	imgAuxiliar[:,:,2] = imgAuxiliar[:,:,0]

	return imgAuxiliar

def histEq(img):

	img_auxiliar = np.copy(img)

	H = np.zeros(256)
	pmf = np.zeros(256) # probability mass function
	cdf = np.zeros(256) # cumulative distributive function
	newLevel = np.zeros(256) # cumulative distributive function

	height, width, channels =  img.shape
	pixels = width * height

	for j in range(0, height):
		for i in  range(0, width):
			H[img[j,i,0]] = H[img[j,i,0]] + 1

	pmf = H / pixels

	for i in range(0, len(pmf)):
		cdf[i] = cdf[i-1] + pmf[i]

	newLevel = cdf * (256 - 1)

	for j in range(0, height):
		for i in range(0, width):
			img_auxiliar[j,i,0] = newLevel[img[j,i,0]]

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def medianFilter(img, n):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)
	#orden = []

	for k in range (0, n):
		for j in range(0, height-1):
			for i in range(0, width-1):
				array = [   img[j-1, i-1, 0], img[j-1, i, 0], img[j-1, i+1, 0],
							  img[j, i-1, 0], 	img[j, i, 0], img[j, i+1, 0],
							img[j+1, i-1, 0], img[j+1, i, 0], img[j+1, i+1, 0]
						]

				array.sort()

				img_auxiliar[j, i, 0] = array[3]

		img[:,:,0] = img_auxiliar[:,:,0]
		img[:,:,1] = img_auxiliar[:,:,0]
		img[:,:,2] = img_auxiliar[:,:,0]

	return img

def meanFilter(img, n):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for k in range (0, n):
		for j in range(0, height-1):
			for i in range(0, width-1):
				list = (   img[j-1, i-1, 0], img[j-1, i, 0], img[j-1, i+1, 0],
							  img[j, i-1, 0], 	img[j, i, 0], img[j, i+1, 0],
							img[j+1, i-1, 0], img[j+1, i, 0], img[j+1, i+1, 0]
						)

				suma = sum(list)

				img_auxiliar[j, i, 0] = 0.1111 * suma

		img[:,:,0] = img_auxiliar[:,:,0]
		img[:,:,1] = img_auxiliar[:,:,0]
		img[:,:,2] = img_auxiliar[:,:,0]

	return img

def firstDerivativeX(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (	0*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	0*img[j-1, i+1, 0],
						0*img[j, i-1, 0], 		-1*img[j, i, 0], 	1*img[j, i+1, 0],
						0*img[j+1, i-1, 0], 	0*img[j+1, i, 0], 	0*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0:
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def firstDerivativeY(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (	0*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	0*img[j-1, i+1, 0],
						0*img[j, i-1, 0], 		-1*img[j, i, 0], 	0*img[j, i+1, 0],
						0*img[j+1, i-1, 0], 	1*img[j+1, i, 0], 	0*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0:
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def gradiente(img):

	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):

			listx = (	0*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	0*img[j-1, i+1, 0],
						0*img[j, i-1, 0], 		-1*img[j, i, 0], 	1*img[j, i+1, 0],
						0*img[j+1, i-1, 0], 	0*img[j+1, i, 0], 	0*img[j+1, i+1, 0]
					)

			sumaX = sum(listx)

			listy = (	0*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	0*img[j-1, i+1, 0],
						0*img[j, i-1, 0], 		-1*img[j, i, 0], 	0*img[j, i+1, 0],
						0*img[j+1, i-1, 0], 	1*img[j+1, i, 0], 	0*img[j+1, i+1, 0]
					)

			sumaY = sum(listy)

			suma = sumaX**2 + sumaY**2
			modulo = math.sqrt(suma)

			if suma > 0 :
				img_auxiliar[j, i, 0] = modulo
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def secondDerivativeX(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height):
		for i in range(0, width-1):
			if img[j, i+1, 0] + img[j, i-1, 0] > 2*img[j, i, 0] :
				img_auxiliar[j, i, 0] = img[j, i+1, 0] + img[j, i-1, 0] - 2*img[j, i, 0]
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def secondDerivativeY(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width):
			if img[j+1, i, 0] + img[j-1, i, 0] > 2*img[j, i, 0] :
				img_auxiliar[j, i, 0] = img[j+1, i, 0] + img[j-1, i, 0] - 2*img[j, i, 0]
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def bordesHorizontal(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (	-1*img[j-1, i-1, 0], 	-1*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
						 2*img[j, i-1, 0], 		 2*img[j, i, 0], 	 2*img[j, i+1, 0],
						-1*img[j+1, i-1, 0], 	-1*img[j+1, i, 0], 	-1*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0 :
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def bordesVertical(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (   	-1*img[j-1, i-1, 0], 	2*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
						-1*img[j, i-1, 0], 		2*img[j, i, 0], 	-1*img[j, i+1, 0],
						-1*img[j+1, i-1, 0], 	2*img[j+1, i, 0], 	-1*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0 :
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def bordesP45(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (   	-1*img[j-1, i-1, 0], 	-1*img[j-1, i, 0], 	 2*img[j-1, i+1, 0],
						-1*img[j, i-1, 0], 		 2*img[j, i, 0], 	-1*img[j, i+1, 0],
						 2*img[j+1, i-1, 0], 	-1*img[j+1, i, 0], 	-1*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0 :
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def bordesN45(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (   	 2*img[j-1, i-1, 0], 	-1*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
						-1*img[j, i-1, 0], 		 2*img[j, i, 0], 	-1*img[j, i+1, 0],
						-1*img[j+1, i-1, 0], 	-1*img[j+1, i, 0], 	 2*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0 :
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def sobelX(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (   	-1*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	1*img[j-1, i+1, 0],
						-2*img[j, i-1, 0], 		0*img[j, i, 0], 	2*img[j, i+1, 0],
						-1*img[j+1, i-1, 0], 	0*img[j+1, i, 0], 	1*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0 :
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def sobelY(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):
			list = (   	-1*img[j-1, i-1, 0], 	-2*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
						 0*img[j, i-1, 0], 		 0*img[j, i, 0], 	 0*img[j, i+1, 0],
						 1*img[j+1, i-1, 0], 	 2*img[j+1, i, 0], 	 1*img[j+1, i+1, 0]
					)

			suma = sum(list)

			if suma > 0 :
				img_auxiliar[j, i, 0] = suma
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def sobelXY(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):

			listx = (   -1*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	1*img[j-1, i+1, 0],
						-2*img[j, i-1, 0], 		0*img[j, i, 0], 	2*img[j, i+1, 0],
						-1*img[j+1, i-1, 0], 	0*img[j+1, i, 0], 	1*img[j+1, i+1, 0]
					)

			sumaX = sum(listx)

			listy = (   -1*img[j-1, i-1, 0], 	-2*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
						 0*img[j, i-1, 0], 		 0*img[j, i, 0], 	 0*img[j, i+1, 0],
						 1*img[j+1, i-1, 0], 	 2*img[j+1, i, 0], 	 1*img[j+1, i+1, 0]
					)

			sumaY = sum(listy)

			suma = sumaX**2 + sumaY**2
			modulo = math.sqrt(suma)

			if suma > 0 :
				img_auxiliar[j, i, 0] = modulo
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def prewittXY(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):

			listx = (   -1*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	1*img[j-1, i+1, 0],
						-1*img[j, i-1, 0], 		0*img[j, i, 0], 	1*img[j, i+1, 0],
						-1*img[j+1, i-1, 0], 	0*img[j+1, i, 0], 	1*img[j+1, i+1, 0]
					)

			sumaX = sum(listx)

			listy = (   -1*img[j-1, i-1, 0], 	-1*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
						 0*img[j, i-1, 0], 		 0*img[j, i, 0], 	 0*img[j, i+1, 0],
						 1*img[j+1, i-1, 0], 	 1*img[j+1, i, 0], 	 1*img[j+1, i+1, 0]
					)

			sumaY = sum(listy)

			suma = sumaX**2 + sumaY**2
			magnitud = math.sqrt(suma)

			if suma > 0 :
				img_auxiliar[j, i, 0] = magnitud
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def robertsXY(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):

			listx = (   0*img[j-1, i-1, 0], 	0*img[j-1, i, 0], 	 0*img[j-1, i+1, 0],
						0*img[j, i-1, 0], 		1*img[j, i, 0], 	 0*img[j, i+1, 0],
						0*img[j+1, i-1, 0], 	0*img[j+1, i, 0], 	-1*img[j+1, i+1, 0]
					)

			sumaX = sum(listx)

			listy = (    0*img[j-1, i-1, 0], 	 0*img[j-1, i, 0], 	 0*img[j-1, i+1, 0],
						 0*img[j, i-1, 0], 		 0*img[j, i, 0], 	 1*img[j, i+1, 0],
						 0*img[j+1, i-1, 0], 	-1*img[j+1, i, 0], 	 0*img[j+1, i+1, 0]
					)

			sumaY = sum(listy)

			suma = sumaX**2 + sumaY**2
			magnitud = math.sqrt(suma)

			if suma > 0 :
				img_auxiliar[j, i, 0] = magnitud
			else:
				img_auxiliar[j, i, 0] = 0

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def kirsch(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):

			K0 = [	-3*img[j-1, i-1, 0],	-3*img[j-1, i, 0], 	 5*img[j-1, i+1, 0],
					-3*img[j, i-1, 0], 		 0*img[j, i, 0], 	 5*img[j, i+1, 0],
					-3*img[j+1, i-1, 0], 	-3*img[j+1, i, 0],	 5*img[j+1, i+1, 0] ]

			K1 = [ 	-3*img[j-1, i-1, 0],	 5*img[j-1, i, 0], 	 5*img[j-1, i+1, 0],
					-3*img[j, i-1, 0], 		 0*img[j, i, 0], 	 5*img[j, i+1, 0],
					-3*img[j+1, i-1, 0], 	-3*img[j+1, i, 0], 	-3*img[j+1, i+1, 0] ]

			K2 = [ 	 5*img[j-1, i-1, 0],	 5*img[j-1, i, 0], 	 5*img[j-1, i+1, 0],
					-3*img[j, i-1, 0], 		 0*img[j, i, 0],   	-3*img[j, i+1, 0],
					-3*img[j+1, i-1, 0], 	-3*img[j+1, i, 0],	-3*img[j+1, i+1, 0] ]

			K3 = [ 	 5*img[j-1, i-1, 0],	 5*img[j-1, i, 0], 	-3*img[j-1, i+1, 0],
					 5*img[j, i-1, 0], 		 0*img[j, i, 0], 	-3*img[j, i+1, 0],
					-3*img[j+1, i-1, 0], 	-3*img[j+1, i, 0],	-3*img[j+1, i+1, 0] ]

			K4 = [ 	 5*img[j-1, i-1, 0],	-3*img[j-1, i, 0], 	-3*img[j-1, i+1, 0],
					 5*img[j, i-1, 0], 		 0*img[j, i, 0], 	-3*img[j, i+1, 0],
					 5*img[j+1, i-1, 0], 	-3*img[j+1, i, 0],	-3*img[j+1, i+1, 0] ]

			K5 = [ 	-3*img[j-1, i-1, 0],	-3*img[j-1, i, 0], 	-3*img[j-1, i+1, 0],
					 5*img[j, i-1, 0], 		 0*img[j, i, 0], 	-3*img[j, i+1, 0],
					 5*img[j+1, i-1, 0], 	 5*img[j+1, i, 0],	-3*img[j+1, i+1, 0] ]

			K6 = [ 	-3*img[j-1, i-1, 0],	-3*img[j-1, i, 0], 	-3*img[j-1, i+1, 0],
					-3*img[j, i-1, 0], 		 0*img[j, i, 0], 	-3*img[j, i+1, 0],
					 5*img[j+1, i-1, 0], 	 5*img[j+1, i, 0],	 5*img[j+1, i+1, 0] ]

			K7 = [ 	-3*img[j-1, i-1, 0],	-3*img[j-1, i, 0], 	-3*img[j-1, i+1, 0],
					-3*img[j, i-1, 0], 		 0*img[j, i, 0], 	 5*img[j, i+1, 0],
					-3*img[j+1, i-1, 0], 	 5*img[j+1, i, 0],	 5*img[j+1, i+1, 0] ]

			valor = [ sum(K0), sum(K1), sum(K2), sum(K3), sum(K4), sum(K5), sum(K6), sum(K7) ]


			img_auxiliar[j, i, 0] = math.sqrt(max(valor))

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def robinson(img):
	height, width, channels =  img.shape
	img_auxiliar = np.copy(img)

	for j in range(0, height-1):
		for i in range(0, width-1):

			r0 = [	-1*img[j-1, i-1, 0],	 0*img[j-1, i, 0], 	 1*img[j-1, i+1, 0],
					-2*img[j, i-1, 0], 		 0*img[j, i, 0], 	 2*img[j, i+1, 0],
					-1*img[j+1, i-1, 0], 	 0*img[j+1, i, 0],	 1*img[j+1, i+1, 0] ]

			r1 = [ 	 0*img[j-1, i-1, 0],	 1*img[j-1, i, 0], 	 2*img[j-1, i+1, 0],
					-1*img[j, i-1, 0], 		 0*img[j, i, 0], 	 1*img[j, i+1, 0],
					-2*img[j+1, i-1, 0], 	-1*img[j+1, i, 0], 	 0*img[j+1, i+1, 0] ]

			r2 = [ 	 1*img[j-1, i-1, 0],	 2*img[j-1, i, 0], 	 1*img[j-1, i+1, 0],
					 0*img[j, i-1, 0], 		 0*img[j, i, 0],   	 0*img[j, i+1, 0],
					-1*img[j+1, i-1, 0], 	-2*img[j+1, i, 0],	-1*img[j+1, i+1, 0] ]

			r3 = [ 	 2*img[j-1, i-1, 0],	 1*img[j-1, i, 0], 	 0*img[j-1, i+1, 0],
					 1*img[j, i-1, 0], 		 0*img[j, i, 0], 	-1*img[j, i+1, 0],
					 0*img[j+1, i-1, 0], 	-1*img[j+1, i, 0],	-2*img[j+1, i+1, 0] ]

			r4 = [ 	 1*img[j-1, i-1, 0],	 0*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
					 2*img[j, i-1, 0], 		 0*img[j, i, 0], 	-2*img[j, i+1, 0],
					 1*img[j+1, i-1, 0], 	 0*img[j+1, i, 0],	-1*img[j+1, i+1, 0] ]

			r5 = [ 	 0*img[j-1, i-1, 0],	-1*img[j-1, i, 0], 	-2*img[j-1, i+1, 0],
					 1*img[j, i-1, 0], 		 0*img[j, i, 0], 	-1*img[j, i+1, 0],
					 2*img[j+1, i-1, 0], 	 1*img[j+1, i, 0],	 0*img[j+1, i+1, 0] ]

			r6 = [ 	-1*img[j-1, i-1, 0],	-2*img[j-1, i, 0], 	-1*img[j-1, i+1, 0],
					 0*img[j, i-1, 0], 		 0*img[j, i, 0], 	 0*img[j, i+1, 0],
					 1*img[j+1, i-1, 0], 	 2*img[j+1, i, 0],	 1*img[j+1, i+1, 0] ]

			r7 = [ 	-2*img[j-1, i-1, 0],	-1*img[j-1, i, 0], 	 0*img[j-1, i+1, 0],
					-1*img[j, i-1, 0], 		 0*img[j, i, 0], 	 1*img[j, i+1, 0],
					 0*img[j+1, i-1, 0], 	 1*img[j+1, i, 0],	 2*img[j+1, i+1, 0] ]

			valor = [ sum(r0), sum(r1), sum(r2), sum(r3), sum(r4), sum(r5), sum(r6), sum(r7) ]


			img_auxiliar[j, i, 0] = math.sqrt(max(valor))

	img[:,:,0] = img_auxiliar[:,:,0]
	img[:,:,1] = img_auxiliar[:,:,0]
	img[:,:,2] = img_auxiliar[:,:,0]

	return img

def funcionDistancia(img):
	height, width, channels = img.shape
	imgAuxiliar = np.copy(img)

	#imgAuxiliar = umbralGrises(img, 127, 255)

	imgAuxiliar[0,:,0] = 0
	imgAuxiliar[:,0,0] = 0
	imgAuxiliar[height-1,:,0] = 0
	imgAuxiliar[:,width-1,0] = 0

	for j in range(1, height):
		for i in range(1, width-1):
			if imgAuxiliar[j, i, 0] == 255:
				list = ( imgAuxiliar[j-1, i-1, 0], imgAuxiliar[j-1, i, 0], imgAuxiliar[j-1, i+1, 0],
						 imgAuxiliar[j, i-1, 0])
				valor = 1 + min(list)

				imgAuxiliar[j, i, 0] = valor

	for j in range(height-2, -1, -1):
		for i in range(width-2, 0, -1):
			if imgAuxiliar[j, i, 0] != 0:
				list = ( 			       				   					imgAuxiliar[j, i+1, 0],
						 imgAuxiliar[j+1, i-1, 0], imgAuxiliar[j+1, i, 0], 	imgAuxiliar[j+1, i+1, 0] )
				valor = min(imgAuxiliar[j, i, 0], 1 + min(list))

				imgAuxiliar[j, i, 0] = valor

	imgAuxiliar[:,:,1] = imgAuxiliar[:,:,0]
	imgAuxiliar[:,:,2] = imgAuxiliar[:,:,0]

	return imgAuxiliar

def ihsl2rgb(h,s,l):
	heigh, widht, channels = h.shape
	imgRGB = np.zeros((256,256,3), dtype=np.uint8)

	h = h * 360 / 255
	print h

	return imgRGB
