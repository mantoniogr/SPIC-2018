#!/usr/bin/python

import mmdepth as mm
import numpy as np
import cv2

import glob

# Ask user for folder
# directory_name = raw_input("-> Directory name: ")
path = "../images/"

# Read depth images from database folder
images = glob.glob(path + '/*.png')
print (images)

for fname in images:
	# filter size
	# kernel = 2*size + 1
	size = 1

	# Read image
	imgOriginal = cv2.imread(fname)
	# imgOriginal = cv2.imread("cones-disp2.png")

	# Image negative
	# img = mm.negativo(imgOriginal)
	# cv2.imwrite("negativo.png", img)

	img = np.copy(imgOriginal)

	# Count white pixels
	noise = mm.noiseCount(img)

	# Opening size 3x3
	imgProcesada = mm.closing(img, size)
	cv2.imwrite(fname[0:len(fname)-4] + "-closing.png", imgProcesada)
	filtered = np.copy(imgProcesada)
	# Count white pixels
	noise = mm.noiseCount(imgProcesada)

	while(noise):
		imgProcesada, marcador = mm.cerraduraReconstruccionM(imgProcesada, size)
		# cv2.imwrite("Cerradura R" + str(size) + ".png", imgProcesada)
		# cv2.imwrite("Marcador R" + str(size) + ".png", marcador)
		noise = mm.noiseCount(imgProcesada)
		size = size + 1
		filtered = mm.highPass(filtered, imgProcesada)

	# filtered = mm.negativo(filtered)
	cv2.imwrite(fname[0:len(fname)-4] + "-MCbR.png", filtered)
