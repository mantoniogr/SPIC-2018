import cv2
import numpy as np
import math
import functions as f
import morphology as mm

def etiquetado(img):
	imgAuxiliar = np.copy(img)
	img2 = np.copy(img)

	height, width, channels =  img.shape

	k = 0
	l = 10
	fifo = []

	lista = []
	for i in range(0, 3000):
		lista.append(i)
	# # fifo jerarquica
	fifoj = {key: 0 for key in lista}

	imgAuxiliar[0,:,0] = 0
	imgAuxiliar[:,0,0] = 0
	imgAuxiliar[height-1,:,0] = 0
	imgAuxiliar[:,width-1,0] = 0

	for j in range(0, height):
		for i in range(0, width):
			if imgAuxiliar[j,i,0] != 0:
				k = k + 5
				l = l + 1
				fifo.append([j,i])
				imgAuxiliar[j,i,0] = 0
				img2[j,i,0] = k
				fifoj[k] += 1
				while(fifo):
					primas = fifo.pop(0)
					for n in range(primas[0] - 1, primas[0] + 2):
						for m in range(primas[1] - 1, primas[1] + 2):
							if imgAuxiliar[n,m,0] != 0:
								fifo.append([n,m])
								imgAuxiliar[n,m,0] = 0
								img2[n,m,0] = k
								fifoj[k] += 1

	# print k
	# print fifoj
	# suma = 0

	# for key in fifoj:
	# 	print fifoj[key]
	# 	suma = fifoj[key] + suma

	# print "Suma: " + str(suma)
	# print "Promedio: " + str(suma/k)

	imgAuxiliar[:,:,1] = imgAuxiliar[:,:,0]
	imgAuxiliar[:,:,2] = imgAuxiliar[:,:,0]

	img2[:,:,1] = img2[:,:,0]
	img2[:,:,2] = img2[:,:,0]

	return img2

def watershed(ime, minimos, original):
	height, width, channels = ime.shape
	fp = np.copy(ime)
	gp = np.copy(ime)
	#ims = np.copy(ime) # watershed
	ims = original # watershed
	#mask = minimos(ime) # minimos de ime
	mask = minimos # minimos de ime
	imwl = etiquetado(mask) # vertientes

	# cv2.imshow("etiq", mask)

	imwl[0,:,0] = 0
	imwl[:,0,0] = 0
	imwl[height-1,:,0] = 0
	imwl[:,width-1,0] = 0

	lista = []
	for i in range(0, 256):
		lista.append(i)
	# # fifo jerarquica
	fifoj = {key: [] for key in lista}

	for j in range(1, height-1):
		for i in range(1, width-1):
			if imwl[j,i,0] != 0:
				ban_ = 255
				for k in range(j-1, j+2):
					for l in range(i-1, i+2):
						ban_ = ban_ & imwl[k,l,0]
				if ban_ == 0:
					fifoj[ime[j,i,0]].append([j,i])

	i = 0
	while(i!=256):
		while(bool(fifoj[i]) is True):
			coord = fifoj[i].pop(0)

			for k in range(coord[0]-1, coord[0]+2):
				for l in range(coord[1]-1, coord[1]+2):
					if l < width and k < height and l > 0 and k > 0 and imwl[k,l,0] == 0:
						for n in range(k-1, k+2):
							for m in range(l-1, l+2):
								if m < width and n < height and m > 0 and n > 0 and imwl[n,m,0] != imwl[coord[0],coord[1],0] and imwl[n,m,0] != 0 and imwl[n,m,0] != 1000000:
									ims[k,l,0] = 255
								imwl[k,l,0] = imwl[coord[0],coord[1],0]
								fifoj[ime[k,l,0]].append([k,l])
		i = i + 1

	ims[:,:,1] = ims[:,:,0]
	ims[:,:,2] = ims[:,:,0]

	imwl[:,:,1] = imwl[:,:,0]
	imwl[:,:,2] = imwl[:,:,0]

	return ims, imwl

def color_segmentation(img, type):

	# img = cv2.imread("../Image4/im6.png")

	height, width , channels = img.shape

	matL = np.copy(img)
	matS = np.copy(img)
	matH = np.copy(img)

	imgAuxiliar = np.zeros((height, width, 3), dtype=np.float64)

	for j in range(0, height):
		for i in range(0, width):
			imgAuxiliar[j,i,0] = img[j,i,0] / 255.0
			imgAuxiliar[j,i,1] = img[j,i,1] / 255.0
			imgAuxiliar[j,i,2] = img[j,i,2] / 255.0

	matLN = np.copy(imgAuxiliar)
	matSN = np.copy(imgAuxiliar)
	matHN = np.copy(imgAuxiliar)

	for j in range(0, height):
		for i in range(0, width):
			# bgr
			matLN[j,i,0] = (0.213 * imgAuxiliar[j,i,2]) + (0.715 * imgAuxiliar[j,i,1]) + (0.072 * imgAuxiliar[j,i,0])
			# print matLN[j,i,0]

			matSN[j,i,0] = max(imgAuxiliar[j,i,2], imgAuxiliar[j,i,1], imgAuxiliar[j,i,0]) - min(imgAuxiliar[j,i,2], imgAuxiliar[j,i,1], imgAuxiliar[j,i,0])
			# print matSN[j,i,0]

			if imgAuxiliar[j,i,2] > imgAuxiliar[j,i,1] and imgAuxiliar[j,i,2] > imgAuxiliar[j,i,0]:
				matHN[j,i,0] = (imgAuxiliar[j,i,1] - imgAuxiliar[j,i,0]) / matSN[j,i,0]
				matHN[j,i,0] = (matHN[j,i,0]*60.0)
				# print matH[j,i,0]

			elif imgAuxiliar[j,i,1] > imgAuxiliar[j,i,0] and imgAuxiliar[j,i,1] > imgAuxiliar[j,i,2]:
				matHN[j,i,0] = ((imgAuxiliar[j,i,0] - imgAuxiliar[j,i,2]) / matSN[j,i,0]) + 2.0
				matHN[j,i,0] = (matHN[j,i,0]*60.0)
				# print matH[j,i,0]

			elif imgAuxiliar[j,i,0] > imgAuxiliar[j,i,1] and imgAuxiliar[j,i,0] > imgAuxiliar[j,i,2]:
				matHN[j,i,0] = ((imgAuxiliar[j,i,2] - imgAuxiliar[j,i,1]) / matSN[j,i,0]) + 4.0
				matHN[j,i,0] = (matHN[j,i,0]*60.0)
				# print matH[j,i,0]
			else:
				matHN[j,i,0] = 0
				matHN[j,i,0] = (matHN[j,i,0]*60.0)

	# Desplegar HSL
	#
	for j in range(0, height):
		for i in range(0, width):
			matS[j,i,0] = matSN[j,i,0] * 255.0
			matL[j,i,0] = matLN[j,i,0] * 255.0
			matH[j,i,0] = matHN[j,i,0] * 255.0 / 360.0

	matS[:,:,1] = matS[:,:,0]
	matS[:,:,2] = matS[:,:,0]

	matH[:,:,1] = matH[:,:,0]
	matH[:,:,2] = matH[:,:,0]

	matL[:,:,1] = matL[:,:,0]
	matL[:,:,2] = matL[:,:,0]

	#cv2.imshow("S", matS)
	#cv2.imshow("H", matH)
	#cv2.imshow("L", matL)
	# #
	#

	HS = np.zeros((256, 256, channels), dtype=np.uint16)
	HSlog = np.zeros((256, 256, channels), dtype=np.float64)
	LS = np.zeros((256, 256, channels), dtype=np.uint16)
	LSlog = np.zeros((256, 256, channels), dtype=np.float64)

	for j in range(0,height):
		for i in range(0,width):

			# HS[matH[j,i,0],matS[j,i,0],0] = HS[matH[j,i,0],matS[j,i,0],0] + 1
			y = (matS[j,i,0]/2) * math.sin( (matH[j,i,0] * 360.0 / 255.0) * (math.pi / 180.0) )
			x = (matS[j,i,0]/2) * math.cos( (matH[j,i,0] * 360.0 / 255.0) * (math.pi / 180.0) )
			xf = 127 + x
			yf = 127 + y
			HS[int(yf),int(xf),0] = HS[int(yf),int(xf),0] + 1

			# LS[matL[j,i,0],matS[j,i,0],0] = LS[matL[j,i,0],matS[j,i,0],0] + 1
			LS[matS[j,i,0],matL[j,i,0],0] = LS[matS[j,i,0],matL[j,i,0],0] + 1

	for j in range(0,256):
		for i in range(0,256):
			if HS[j,i,0] > 0:
				HSlog[j,i,0] = math.log10(HS[j,i,0])
			if LS[j,i,0] > 0:
				LSlog[j,i,0] = math.log10(LS[j,i,0])
				# print LSlog[j,i,0]

	maxLSlog = 0.0
	maxHSlog = 0.0

	for j in range(0,256):
		for i in range(0,256):
			if HSlog[j,i,0] > maxHSlog:
				maxHSlog = HSlog[j,i,0]
			if LSlog[j,i,0] > maxLSlog:
				maxLSlog = LSlog[j,i,0]

	# print maxHSlog
	# print maxLSlog

	HSF = np.zeros((256, 256, channels), dtype=np.uint8)
	LSF = np.zeros((256, 256, channels), dtype=np.uint8)

	for j in range(0,256):
		for i in range(0,256):
			HSF[j,i,0] = (HSlog[j,i,0] / maxHSlog) * 255
			LSF[j,i,0] = (LSlog[j,i,0] / maxLSlog) * 255

	HSF[:,:,1] = HSF[:,:,0]
	HSF[:,:,2] = HSF[:,:,0]

	LSF[:,:,1] = LSF[:,:,0]
	LSF[:,:,2] = LSF[:,:,0]

	#cv2.imshow("HS", HSF)
	#cv2.imshow("LS", LSF)
	# cv2.imwrite("0HS.png", HSF)
	# cv2.imwrite("0LS.png", LSF)

	# # LS
	if(type=='ls'):
		cerraduraLS = mm.cerradura(LSF,2)
		#cv2.imshow("Cerradura LS", cerraduraLS)
		# cv2.imwrite("1CerraduraLS.png", cerraduraLS)

		negativoLS = f.negativoGrises(cerraduraLS)
		#cv2.imshow("Cerradura negative LS", negativoLS)
		# cv2.imwrite("2NegativoLS.png", negativoLS)

		alternadoLS = mm.secuencialRecontruccion1(negativoLS, 1)
		#cv2.imshow("Alternado LS", alternadoLS)
		# cv2.imwrite("3AlternadoLS.png", alternadoLS)

		minimosLS = mm.minimos(alternadoLS)
		#cv2.imshow("LS negative minimos", minimosLS)
		# cv2.imwrite("4MinimosLS.png", minimosLS)

		minimosLS[0,:,0] = 0
		minimosLS[:,0,0] = 0
		minimosLS[256-1,:,0] = 0
		minimosLS[:,256-1,0] = 0

		minimosLS[:,:,1] = minimosLS[:,:,0]
		minimosLS[:,:,2] = minimosLS[:,:,0]

		imgB = np.zeros((256,256,3), dtype=np.uint8)

		watershedLS, vertientesLS = watershed(alternadoLS, minimosLS, imgB)
		#cv2.imshow("Watershed Black LS", watershedLS)
		#cv2.imshow("Vertientes LS", vertientesLS)

		# cv2.imwrite("5WatershedLS.png", watershedLS)
		# cv2.imwrite("5VertientesLS.png", vertientesLS)

		imgResultanteLS = np.copy(img)

		for j in range(0, height):
			for i in range(0, width):
				imgResultanteLS[j,i,0] = vertientesLS[matS[j,i,0],matL[j,i,0],0]

		imgResultanteLS[:,:,1] = imgResultanteLS[:,:,0]
		imgResultanteLS[:,:,2] = imgResultanteLS[:,:,0]

		imgResultante = mm.aperturaReconstruccion(imgResultanteLS, 3)

		#cv2.imshow("Segmented LS", imgResultanteLS)
		# cv2.imwrite("6ResultanteLS.png", imgResultanteLS)


	# HS
	if(type=='hs'):

		cerraduraHS = mm.cerradura(HSF,2)
		# cv2.imshow("Cerradura HS", cerraduraHS)
		# cv2.imwrite("1CerraduraHS.png", cerraduraHS)

		negativoHS = f.negativoGrises(cerraduraHS)
		# cv2.imshow("Cerradura HS negative", negativoHS)
		# cv2.imwrite("2NegativoHS.png", negativoHS)

		alternadoHS = mm.secuencialRecontruccion1(negativoHS, 1)
		# cv2.imshow("Alternado HS", alternadoHS)
		# cv2.imwrite("3AlternadoHS.png", alternadoHS)

		minimosHS = mm.minimos(alternadoHS)
		# cv2.imshow("HS negative minimos", minimosHS)
		# cv2.imwrite("4MinimosHS.png", minimosHS)

		minimosHS[0,:,0] = 0
		minimosHS[:,0,0] = 0
		minimosHS[256-1,:,0] = 0
		minimosHS[:,256-1,0] = 0

		minimosHS[:,:,1] = minimosHS[:,:,0]
		minimosHS[:,:,2] = minimosHS[:,:,0]

		imgC = np.zeros((256,256,3), dtype=np.uint8)

		watershedHS, vertientesHS = watershed(alternadoHS, minimosHS, imgC)
		# cv2.imshow("Watershed Black HS", watershedHS)
		# cv2.imshow("Vertientes HS", vertientesHS)

		# cv2.imwrite("5WatershedHS.png", watershedHS)
		# cv2.imwrite("5VertientesHS.png", vertientesHS)

		imgResultanteHS = np.copy(img)
		imgRGB = np.copy(img)

		for j in range(0, height):
			for i in range(0, width):
				# imgResultanteHS[j,i,0] = vertientesHS[matH[j,i,0],matS[j,i,0],0]
				# imgResultanteHS[j,i,0] = vertientesHS[matS[j,i,0],matH[j,i,0],0]

				y = (matS[j,i,0]/2) * math.sin( (matH[j,i,0] * 360.0 / 255.0) * (math.pi / 180.0) )
				x = (matS[j,i,0]/2) * math.cos( (matH[j,i,0] * 360.0 / 255.0) * (math.pi / 180.0) )
				xf = 127 + x
				yf = 127 + y

				imgResultanteHS[j,i,0] = vertientesHS[int(yf), int(xf), 0]

		imgResultanteHS[:,:,1] = imgResultanteHS[:,:,0]
		imgResultanteHS[:,:,2] = imgResultanteHS[:,:,0]

		imgResultante = mm.aperturaReconstruccion(imgResultanteHS, 3)

		# cv2.imshow("Segmented HS", imgResultanteHS)
		# cv2.imwrite("6ResultanteHS.png", imgResultanteHS)

		#cv2.waitKey(0)

	return imgResultante

def color_assignation(imagenColor, imagenSegmentada):
	# imagenColor = cv2.imread("../Image4/im6.png")
	# imagenSegmentada = cv2.imread("6ResultanteHS.png")

	height, width, channels = imagenColor.shape

	lista = []

	for i in range(0, 256):
		lista.append(i)

	# #
	b = {key: 0 for key in lista}
	g = {key: 0 for key in lista}
	r = {key: 0 for key in lista}

	b_counter = {key: 0 for key in lista}
	g_counter = {key: 0 for key in lista}
	r_counter = {key: 0 for key in lista}

	for j in range(0, height):
		for i in range(0, width):
			b[imagenSegmentada[j,i,0]] = b[imagenSegmentada[j,i,0]] + imagenColor[j,i,0]
			b_counter[imagenSegmentada[j,i,0]] = b_counter[imagenSegmentada[j,i,0]] + 1

			g[imagenSegmentada[j,i,1]] = g[imagenSegmentada[j,i,1]] + imagenColor[j,i,1]
			g_counter[imagenSegmentada[j,i,0]] = g_counter[imagenSegmentada[j,i,0]] + 1

			r[imagenSegmentada[j,i,2]] = r[imagenSegmentada[j,i,2]] + imagenColor[j,i,2]
			r_counter[imagenSegmentada[j,i,0]] = r_counter[imagenSegmentada[j,i,0]] + 1

	counter = 0
	for i in range(0, 256):
		if b[i] != 0:
			counter = counter + 1

	for i in range(0, 256):
		if b_counter[i] != 0:
			b[i] = b[i] / b_counter[i]
		if g_counter[i] != 0:
			g[i] = g[i] / g_counter[i]
		if r_counter[i] != 0:
			r[i] = r[i] / r_counter[i]

	imagenFalsoColor = np.copy(imagenSegmentada)

	for j in range(0, height):
		for i in range(0, width):
			imagenFalsoColor[j,i,0] = b[imagenSegmentada[j,i,0]]
			imagenFalsoColor[j,i,1] = g[imagenSegmentada[j,i,0]]
			imagenFalsoColor[j,i,2] = r[imagenSegmentada[j,i,0]]

	#cv2.imshow("Falso Color", imagenFalsoColor)
	# cv2.imwrite("7 FalsoColorLS.png", imagenFalsoColor)
	#cv2.waitKey(0)

	return imagenFalsoColor
