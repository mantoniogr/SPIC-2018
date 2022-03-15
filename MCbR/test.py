#!/usr/bin/python

import mmdepth as mm
import numpy as np
import cv2

def noiseCount(img):
	height, width, channels =  img.shape
	
	counter = 0
	
	for j in range(0,height):
		for i in range(0, width):
			if (img[j,i,0] == 0):
				counter = 1 + counter
	
	print (counter)
	return counter

size = 1

imgOriginal = cv2.imread("../images/disp2-barn1-noise-closing.png")
height, width, channels =  imgOriginal.shape

imgProcesada = np.copy(imgOriginal)

noise = noiseCount(imgProcesada)

# imgProcesada = mm.dilation(imgProcesada, 5)
# cv2.imwrite("dilation " + str(5) + ".png", imgProcesada)
# noise = noiseCount(imgProcesada)

while(noise):
	# imgProcesada = cv2.medianBlur(imgOriginal, size)
	# imgProcesada = mm.dilation(imgOriginal, size)
	# imgProcesada = mm.closing(imgOriginal, size)
	# imgProcesada = mm.secuencial2(imgOriginal, size)
	# imgProcesada, marker = mm.cerraduraReconstruccion(imgOriginal, size)
	imgProcesada, marker = mm.cerraduraReconstruccionM(imgOriginal, size)
	cv2.imwrite("secuencial2 " + str(size) + ".png", imgProcesada)
	noise = noiseCount(imgProcesada)
	size = size + 1