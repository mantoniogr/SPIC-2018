# libraries
import numpy as np
import cv2

def dilation(map, size):
    height, width =  map.shape
    auxMap = np.copy(map)

    for k in range(0, size):
        # B1
        for j in range(0, height):
            for i in range(0, width-1):
                if auxMap[j, i] < auxMap[j, i+1]:
                    auxMap[j, i] = auxMap[j, i+1]

        # B2
        for j in range(0, height-1):
            for i in range(0, width):
                if auxMap[j, i] < auxMap[j+1, i]:
                    auxMap[j, i] = auxMap[j+1, i]

        # B3
        for j in range(0, height):
            for i in range(width-1, 0, -1):
                if auxMap[j, i] < auxMap[j, i-1]:
                    auxMap[j, i] = auxMap[j, i-1]

        # B4
        for j in range(height-1, 0, -1):
            for i in range(0, width):
                if auxMap[j, i] < auxMap[j-1, i]:
                    auxMap[j, i] = auxMap[j-1, i]

    #auxMap[:,:,1] = auxMap[:,:,0]
    #auxMap[:,:,2] = auxMap[:,:,0]

    return auxMap

def erosion(map, size):
    height, width =  map.shape
    auxMap = np.copy(map)

    auxMap = negativo(auxMap)
    auxMap = dilation(auxMap,size)
    auxMap = negativo(auxMap)

    return auxMap

def opening(map, size):
    auxMap = np.copy(map)

    auxMap = erosion(auxMap, size)
    auxMap = dilation(auxMap, size)

    return auxMap

def closing(map, size):
    auxMap = np.copy(map)

    auxMap = dilation(auxMap, size)
    auxMap = erosion(auxMap, size)

    return auxMap

def tophatW(map, size):
    height, width, channels =  map.shape
    imgOriginal = np.copy(map)
    auxMap = np.copy(map)

    auxMap = opening(auxMap, size)

    for j in range(0, height):
        for i in range(0, width):
            auxMap[j, i, 0] = imgOriginal[j, i, 0] - auxMap[j, i, 0]

    auxMap[:,:,1] = auxMap[:,:,0]
    auxMap[:,:,2] = auxMap[:,:,0]

    return auxMap

def tophatB(map, size):
    height, width, channels =  map.shape
    imgOriginal = np.copy(map)
    auxMap = np.copy(map)

    auxMap = closing(auxMap, size)

    for j in range(0, height):
        for i in range(0, width):
            auxMap[j, i, 0] = auxMap[j, i, 0] - imgOriginal[j, i, 0]

    auxMap[:,:,1] = auxMap[:,:,0]
    auxMap[:,:,2] = auxMap[:,:,0]

    return auxMap

def secuencial1(map, size):
    auxMap = np.copy(map)

    for i in range(0, size):
        auxMap = opening(auxMap, i+1)
        auxMap = closing(auxMap, i+1)

    return auxMap

def secuencial2(map, size):
    auxMap = np.copy(map)

    for i in range(0, size):
        auxMap = closing(auxMap, i+1)
        auxMap = opening(auxMap, i+1)

    return auxMap

def gradienteMorfologico(map, size):
    auxMap = np.copy(map)
    auxMap2 = np.copy(map)

    auxMap = dilation(auxMap, size)
    auxMap2 = erosion(auxMap2, size)

    auxMap = auxMap - auxMap2

    return auxMap

def gradienteExterno(map, n):
    auxMap = np.copy(map)
    imgOriginal = np.copy(map)

    auxMap = dilation(auxMap, n)
    auxMap = auxMap - imgOriginal

    return auxMap

def gradienteInterno(map, n):
    auxMap = np.copy(map)
    imgOriginal = np.copy(map)

    auxMap = erosion(auxMap, n)
    auxMap = imgOriginal - auxMap

    return auxMap

def dilatacionGeodesica(I, J):
    height, width, channels =  I.shape
    flag = True

    while(flag):

        img_auxiliar = np.copy(J)

        for j in range(1, height):
            for i in range(1, width-1):
                list1 = (   J[j-1, i-1, 0], J[j-1, i, 0],   J[j-1, i+1, 0],
                            J[j, i-1, 0],   J[j, i, 0])
                J[j, i, 0] = min([max(list1), I[j,i,0]])

        for j in range(height-2, -1, -1):
            for i in range(width-2, 0, -1):
                list2 = (                   J[j, i, 0],     J[j, i+1, 0],
                            J[j+1, i-1, 0], J[j+1, i, 0],   J[j+1, i+1, 0] )
                J[j, i, 0] = min([max(list2), I[j, i, 0]])

        dif = J - img_auxiliar

        if np.amax(dif) == 0:
            flag = False

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

def dilatacionGeodesicaM(I, J):
    height, width, channels =  I.shape
    flag = True

    while(flag):

        img_auxiliar = np.copy(J)

        for j in range(1, height):
            for i in range(1, width-1):
                list1 = (   J[j-1, i-1, 0],     J[j-1, i, 0],   J[j-1, i+1, 0],
                            J[j, i-1, 0],       J[j, i, 0])
                J[j, i, 0] = minNoZero([max(list1), I[j,i,0]])

        for j in range(height-2, -1, -1):
            for i in range(width-2, 0, -1):
                list2 = (                       J[j, i, 0],     J[j, i+1, 0],
                            J[j+1, i-1, 0],     J[j+1, i, 0],   J[j+1, i+1, 0] )
                J[j, i, 0] = minNoZero([max(list2), I[j, i, 0]])

        dif = J - img_auxiliar

        if np.amax(dif) == 0:
            flag = False

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

def minNoZero(lista):
    minimo = 255

    # for x in lista:
    #   if x < 255 and x > 0:
    #       minimo = x

    if lista[0] < 255 and lista[0] > 0:
        minimo = lista[0]
    if lista[1] < 255 and lista[1] > 0:
        minimo = lista[1]

    return minimo

def maxNo255(lista):
	maximo = 0
	for x in lista:
		if x > 0 and x < 255:
			maximo = x

	return maximo

def erosionGeodesicaM(I, J):

    I = negativo(I)
    J = negativo(J)

    J = dilatacionGeodesicaM(I,J)

    I = negativo(I)
    J = negativo(J)

    return J

def erosionGeodesicaM2(I, J):
    height, width, channels =  I.shape
    flag = True

    while(flag):

        img_auxiliar = np.copy(J)

        for j in range(1, height):
            for i in range(1, width-1):
                list1 = (   J[j-1, i-1, 0],     J[j-1, i, 0],   J[j-1, i+1, 0],
                            J[j, i-1, 0],       J[j, i, 0])
                J[j, i, 0] = maxNo255([min(list1), I[j,i,0]])

        for j in range(height-2, -1, -1):
            for i in range(width-2, 0, -1):
                list2 = (                       J[j, i, 0],     J[j, i+1, 0],
                            J[j+1, i-1, 0],     J[j+1, i, 0],   J[j+1, i+1, 0] )
                J[j, i, 0] = maxNo255([min(list2), I[j, i, 0]])

        dif = J - img_auxiliar

        if np.amax(dif) == 0:
            flag = False

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

def erosionGeodesica(I, J):

    I = negativo(I)
    J = negativo(J)

    J = dilatacionGeodesica(I,J)

    I = negativo(I)
    J = negativo(J)

    return J

def aperturaReconstruccion(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = erosion(map, n)
    erosionada = np.copy(Y)
    J = dilatacionGeodesica(img_auxiliar, Y)

    return J, erosionada

def cerraduraReconstruccion(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = dilation(map, n)
    dilatada = np.copy(Y)
    J = erosionGeodesica(img_auxiliar, Y)

    return J, dilatada

def aperturaReconstruccionM(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = erosion(map, n)
    erosionada = np.copy(Y)
    J = dilatacionGeodesicaM(img_auxiliar, Y)

    return J, erosionada

def cerraduraReconstruccionM(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = dilation(map, n)
    dilatada = np.copy(Y)
    J = erosionGeodesicaM(img_auxiliar, Y)

    return J, dilatada

def negativo(map):
    height, width =  map.shape
    auxMap = np.copy(map)

    for j in range(0, height):
        for i in range(0, width):
            auxMap[j, i] = 255 - map[j, i]

    #auxMap[:,:,1] = auxMap[:,:,0]
    #auxMap[:,:,2] = auxMap[:,:,0]

    return auxMap

def threshold(map, umbral):
    height, width, channels =  map.shape
    auxMap = np.copy(map)

    for j in range(0, height):
        for i in range(0, width):
            if auxMap[j,i,0] > umbral:
                auxMap[j,i,0] = 255

    auxMap[:,:,1] = auxMap[:,:,0]
    auxMap[:,:,2] = auxMap[:,:,0]

    return auxMap

def noiseCount(img):
    height, width, channels =  img.shape

    counter = 0

    for j in range(0,height):
        for i in range(0, width):
            if (img[j,i,0] == 0):
                counter = 1 + counter

    print counter
    return counter

# def noiseCount(img):
# 	height, width, channels =  img.shape

# 	counter = 0

# 	for j in range(0,height):
# 		for i in range(0, width):
# 			if (img[j,i,0] == 255):
# 				counter = 1 + counter

# 	print counter
# 	return counter

def highPass(f, g):
    height, width, channels =  f.shape
    img_auxiliar = np.copy(f)

    for j in range(0, height):
        for i in range(0, width):
            # if (f[j,i,0] > g[j,i,0] and f[j,i,0] < 255):
            # if (f[j,i,0] < 255):
            if (f[j,i,0] > 0):
                img_auxiliar[j,i,0] = f[j,i,0]
            else:
                img_auxiliar[j,i,0] = g[j,i,0]

    img_auxiliar[:,:,1] = img_auxiliar[:,:,0]
    img_auxiliar[:,:,2] = img_auxiliar[:,:,0]

    return img_auxiliar
