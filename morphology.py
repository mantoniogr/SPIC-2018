import cv2
import numpy as np
import math
import functions as f

#---------------------------------------------------------------------
# Morfologia matematica
#---------------------------------------------------------------------
def dilatacion(img, n):
    height, width, channels =  img.shape
    imgAuxiliar = np.copy(img)

    for k in range(0, n):
        # B1
        for j in range(0, height):
            for i in range(0, width-1):
                if imgAuxiliar[j, i, 0] < imgAuxiliar[j, i+1, 0]:
                    imgAuxiliar[j, i, 0] = imgAuxiliar[j, i+1, 0]

        # B2
        for j in range(0, height-1):
            for i in range(0, width):
                if imgAuxiliar[j, i, 0] < imgAuxiliar[j+1, i, 0]:
                    imgAuxiliar[j, i, 0] = imgAuxiliar[j+1, i, 0]

        # B3
        for j in range(0, height):
            for i in range(width-1, 0, -1):
                if imgAuxiliar[j, i, 0] < imgAuxiliar[j, i-1, 0]:
                    imgAuxiliar[j, i, 0] = imgAuxiliar[j, i-1, 0]

        # B4
        for j in range(height-1, 0, -1):
            for i in range(0, width):
                if imgAuxiliar[j, i, 0] < imgAuxiliar[j-1, i, 0]:
                    imgAuxiliar[j, i, 0] = imgAuxiliar[j-1, i, 0]

    if channels == 3:
        imgAuxiliar[:,:,1] = imgAuxiliar[:,:,0]
        imgAuxiliar[:,:,2] = imgAuxiliar[:,:,0]

    return imgAuxiliar

def erosion(img, n):
    imgAuxiliar = np.copy(img)

    imgAuxiliar = f.negative(imgAuxiliar)
    imgAuxiliar = dilatacion(imgAuxiliar, n)
    imgAuxiliar = f.negative(imgAuxiliar)

    return imgAuxiliar

def apertura(img, n):
    img_auxiliar = np.copy(img)

    img_auxiliar = erosion(img_auxiliar, n)
    img_auxiliar = dilatacion(img_auxiliar, n)

    return img_auxiliar

def cerradura(img, n):
    img_auxiliar = np.copy(img)

    img_auxiliar = dilatacion(img_auxiliar, n)
    img_auxiliar = erosion(img_auxiliar, n)

    return img_auxiliar

def tophatW(img, n):
    height, width, channels =  img.shape

    img_original = np.copy(img)

    img = apertura(img, n)

    for j in range(0, height):
        for i in range(0, width):
            img[j, i, 0] = int(img_original[j, i, 0]) - int(
                img[j, i, 0])

    img[:,:,1] = img[:,:,0]
    img[:,:,2] = img[:,:,0]

    return img

def tophatB(img, n):
    height, width, channels =  img.shape
    imgAuxiliar = np.copy(img)

    img_original = np.copy(img)

    imgAuxiliar = cerradura(imgAuxiliar, n)

    for j in range(0, height):
        for i in range(0, width):
            imgAuxiliar[j, i, 0] = int(imgAuxiliar[j, i, 0]) - int(img_original[j, i, 0])

    imgAuxiliar[:,:,1] = imgAuxiliar[:,:,0]
    imgAuxiliar[:,:,2] = imgAuxiliar[:,:,0]

    return imgAuxiliar

def secuencial1(img, n):
    img_auxiliar = np.copy(img)
    for i in range(1, n+1):
        img_auxiliar = apertura(img_auxiliar, i)
        img_auxiliar = cerradura(img_auxiliar, i)

    img_auxiliar[:,:,1] = img_auxiliar[:,:,0]
    img_auxiliar[:,:,2] = img_auxiliar[:,:,0]

    return img_auxiliar

def secuencial2(img, n):
    img_auxiliar = np.copy(img)
    for i in range(1, n+1):
        img_auxiliar = cerradura(img_auxiliar, i)
        img_auxiliar = apertura(img_auxiliar, i)

    img_auxiliar[:,:,1] = img_auxiliar[:,:,0]
    img_auxiliar[:,:,2] = img_auxiliar[:,:,0]

    return img_auxiliar

def gradienteMorfologico(img, n):
    img_auxiliar = np.copy(img)

    img = dilatacion(img, n)
    img_auxiliar = erosion(img_auxiliar, n)

    img = img - img_auxiliar

    return img

def gradienteExterno(img, n):
    img_auxiliar = np.copy(img)

    img = dilatacion(img, n)
    img = img - img_auxiliar

    return img

def gradienteInterno(img, n):
    img_auxiliar = np.copy(img)

    img = erosion(img, n)
    img = img_auxiliar - img

    return img

def dilatacionGeodesicaBin(I, J):
    height, width, channels =  I.shape

    fifo = []

    for j in range(0, height):
        for i in range(0, width):
            if J[j,i,0] == 255:
                list_a = [ 	J[j-1,i-1,0], J[j-1,i,0], J[j-1,i+1,0],
                            J[j,i-1,0], 			  J[j,i+1,0],
                            J[j+1,i-1,0], J[j+1,i,0], J[j+1,i+1,0]]
                if min(list_a) == 0 and I[j,i,0] == 255:
                        fifo.append([j,i,0])

    while(fifo):
        p = fifo.pop(0)

        q_y = [ p[0]-1,	p[0], p[0]+1,
                p[0]-1,		  p[0]+1,
                p[0]-1,	p[0], p[0]+1]

        q_x = [ p[1]-1,	p[1]-1, p[1]-1,
                p[1],		  	p[1],
                p[1]+1,	p[1]+1, p[1]+1]

        for i in range(0, len(q_x)):
            if J[q_y[i], q_x[i], 0] == 0 and I[q_y[i], q_x[i], 0] == 255:
                J[q_y[i], q_x[i], 0] = 255
                fifo.append([q_y[i],q_x[i],0])

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

def erosionGeodesicaBin(I, J):

    I = negativoGrises(I)
    J = negativoGrises(J)

    J = dilatacionGeodesicaBin(I,J)

    I = negativoGrises(I)
    J = negativoGrises(J)

    return J

def aperturaReconstruccionBin(img, n):
    img_auxiliar = np.copy(img)
    Y = np.copy(img)

    Y = erosion(Y, n)
    cv2.imshow("Marcador", Y)
    J = dilatacionGeodesicaBin(img_auxiliar, Y)

    return J

def cerraduraReconstruccionBin(img, n):
    img_auxiliar = np.copy(img)

    Y = dilatacion(img, n)
    cv2.imshow("Marcador", Y)
    J = erosionGeodesicaBin(img_auxiliar, Y)

    return J

def dilatacionGeodesica(I, J):
    height, width, channels =  I.shape
    flag = True

    while(flag):

        img_auxiliar = np.copy(J)

        for j in range(1, height):
            for i in range(1, width-1):
                list1 = ( 	J[j-1, i-1, 0], 	J[j-1, i, 0], 	J[j-1, i+1, 0],
                             J[j, i-1, 0], 		J[j, i, 0])
                J[j, i, 0] = min(max(list1), I[j,i,0])

        for j in range(height-2, -1, -1):
            for i in range(width-2, 0, -1):
                list2 = ( 						J[j, i, 0], 	J[j, i+1, 0],
                             J[j+1, i-1, 0],		J[j+1, i, 0], 	J[j+1, i+1, 0] )
                J[j, i, 0] = min(max(list2), I[j, i, 0])

        dif = J - img_auxiliar

        if np.amax(dif) == 0:
            flag = False

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

def erosionGeodesica(I, J):

    I = f.negativoGrises(I)
    J = f.negativoGrises(J)

    J = dilatacionGeodesica(I,J)

    I = f.negativoGrises(I)
    J = f.negativoGrises(J)

    return J

def aperturaReconstruccion(img, n):
    img_auxiliar = np.copy(img)
    Y = np.copy(img)

    Y = erosion(Y, n)
    erosionada = np.copy(Y)
    J = dilatacionGeodesica(img_auxiliar, Y)

    return J#, erosionada

def cerraduraReconstruccion(img, n):
    img_auxiliar = np.copy(img)
    Y = np.copy(img)

    Y = dilatacion(img, n)
    dilatada = np.copy(Y)
    J = erosionGeodesica(img_auxiliar, Y)

    return J#, dilatada

def secuencialRecontruccion1(img, n):
    img_auxiliar = np.copy(img)
    for i in range(1, n+1):
        img_auxiliar = aperturaReconstruccion(img_auxiliar, i)
        img_auxiliar = cerraduraReconstruccion(img_auxiliar, i)

    return img_auxiliar

def secuencialRecontruccion2(img, n):
    img_auxiliar = np.copy(img)
    for i in range(1, n+1):
        img_auxiliar = cerraduraReconstruccion(img_auxiliar, i)
        img_auxiliar = aperturaReconstruccion(img_auxiliar, i)

    return img_auxiliar

def maximos(img):
    height, width, channels =  img.shape
    img_auxiliar = np.copy(img)

    for j in range(0, height):
        for i in range(0, width):
            if img_auxiliar[j,i,0] > 0:
                img_auxiliar[j,i,0] = img_auxiliar[j,i,0] - 1

    img_auxiliar = dilatacionGeodesica(img, img_auxiliar)
    img = img - img_auxiliar
    img = f.umbralGrises(img, 1, 255)

    return img

def minimos(img):
    img_auxiliar = np.copy(img)

    img_auxiliar = f.negativoGrises(img_auxiliar)
    img_auxiliar = maximos(img_auxiliar)
#	img_auxiliar = f.negativoGrises(img_auxiliar)

    return img_auxiliar

def reconstruccionMax(I, J):
    height, width, channels =  I.shape
    IM = np.copy(I)

    J = maximos(J)
    cv2.imshow("Maximos", J)

    for j in range(0, height):
        for i in range(0, width):
            if J[j,i,0] == 255:
                IM[j,i,0] = I[j,i,0]
                IM[j,i,1] = I[j,i,1]
                IM[j,i,2] = I[j,i,2]
            else:
                IM[j,i,0] = 0
                IM[j,i,1] = 0
                IM[j,i,2] = 0

    J = np.copy(IM)

    fifo = []

    for j in range(0, height-1):
        for i in range(0, width-1):

            if J[j, i, 0] != 0:

                p = ([j,i])

                q_y = [ p[0]-1,	p[0]-1, p[0]+1,
                        p[0],		  p[0],
                        p[0]+1,	p[0]+1, p[0]+1]

                q_x = [ p[1]-1,	p[1], p[1]+1,
                        p[1]-1,		  	p[1]+1,
                        p[1]-1,	p[1], p[1]+1]

                for k in range(0, len(q_x)):
                    if J[q_y[k], q_x[k], 0] == 0:
                        fifo.append(p)
    print len(fifo)

    while(fifo):
        p = fifo.pop(0)

        q_y = [ p[0]-1,	p[0]-1, p[0]+1,
                p[0],		  p[0],
                p[0]+1,	p[0]+1, p[0]+1]

        q_x = [ p[1]-1,	p[1], p[1]+1,
                p[1]-1,		  	p[1]+1,
                p[1]-1,	p[1], p[1]+1]

        for i in range(0, len(q_x)):
            if J[q_y[i], q_x[i], 0] < J[p[0], p[1], 0] and I[q_y[i], q_x[i], 0] != J[q_y[i], q_x[i], 0]:
                J[q_y[i], q_x[i], 0] = min(J[p[0], p[1], 0], I[q_y[i], q_x[i], 0])
                fifo.append([q_y[i],q_x[i]])

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

def reconstruccionHibrida(I, J):
    height, width, channels =  I.shape

    for j in range(0, height-1):
        for i in range(0, width-1):
            list1 = ( 	J[j-1, i-1, 0], 	J[j-1, i, 0], 	J[j-1, i+1, 0],
                         J[j, i-1, 0], 		J[j, i, 0])
            maximo = max(list1)
            J[j, i, 0] = min(maximo, I[j,i,0])

    for j in range(height-2, 0, -1):
        for i in range(width-2, 0, -1):
            list2 = ( 						J[j, i, 0], 	J[j, i+1, 0],
                         J[j+1, i-1, 0],		J[j+1, i, 0], 	J[j+1, i+1, 0] )
            maximo = max(list2)
            J[j, i, 0] = min(maximo, I[j,i,0])

    fifo = []

    for j in range(0, height-1):
        for i in range(0, width-1):
            p = ([j,i])

            q_y = [ 				p[0]+1,
                    p[0]-1,	p[0], 	p[0]+1]

            q_x = [ 				p[1],
                    p[1]+1,	p[1]+1, p[1]+1]

            for k in range(0, len(q_x)):
                if J[q_y[k], q_x[k], 0] < J[p[0], p[1], 0] and J[q_y[k], q_x[k], 0] < I[q_y[k], q_x[k], 0]:
                    fifo.append(p)

    while (fifo):
        p = fifo.pop(0)

        q_y = [ p[0]-1,	p[0], p[0]+1,
                p[0]-1,		  p[0]+1,
                p[0]-1,	p[0], p[0]+1]

        q_x = [ p[1]-1,	p[1]-1, p[1]-1,
                p[1],		  	p[1],
                p[1]+1,	p[1]+1, p[1]+1]

        for k in range(0, len(q_x)):
            if q_x < width-1 and q_x > 0 and q_y > 0 and q_y < height-1:
                if J[q_y[k],q_x[k],0] < J[p[0], p[1], 0] and I[q_y[k], q_x[k], 0] != J[q_y[k], q_x[k], 0]:
                    J[q_y[k],q_x[k], 0] = min(J[p[0], p[1], 0], I[q_y[k], q_x[k], 0])
                    fifo.append([q_y[k],q_x[k]])

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

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
                k = k + 1
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

    print k
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

def watershed(ime):
    height, width, channels = ime.shape
    fp = np.copy(ime)
    gp = np.copy(ime)
    ims = np.copy(ime) # watershed
    mask = minimos(ime) # minimos de ime
    imwl = etiquetado(mask) # vertientes

    cv2.imshow("etiq", mask)

    imwl[0,:,0] = 1000000
    imwl[:,0,0] = 1000000
    imwl[height-1,:,0] = 1000000
    imwl[:,width-1,0] = 1000000

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
                    if imwl[k,l,0] == 0:
                        for n in range(k-1, k+2):
                            for m in range(l-1, l+2):
                                if imwl[n,m,0] != imwl[coord[0],coord[1],0] and imwl[n,m,0] != 0 and imwl[n,m,0] != 1000000:
                                    ims[k,l,0] = 255
                                imwl[k,l,0] = imwl[coord[0],coord[1],0]
                                fifoj[ime[k,l,0]].append([k,l])
        i = i + 1

    ims[:,:,1] = ims[:,:,0]
    ims[:,:,2] = ims[:,:,0]

    imwl[:,:,1] = imwl[:,:,0]
    imwl[:,:,2] = imwl[:,:,0]

    return ims, imwl

def etiquetado2(img):
    imgAuxiliar = np.copy(img)
    img2 = np.copy(img)

    height, width, channels =  img.shape

    k = 0
    l = 10
    fifo = []

    lista = []
    for i in range(0, 500):
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
                k = k + 1
                l = l + 10
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

    print k
    print fifoj
    suma = 0

    for key in fifoj:
        print fifoj[key]
        suma = fifoj[key] + suma

    print "Suma: " + str(suma)
    print "Promedio: " + str(suma/k)
    print "K:" + str(k)

    imgAuxiliar[:,:,1] = imgAuxiliar[:,:,0]
    imgAuxiliar[:,:,2] = imgAuxiliar[:,:,0]

    img2[:,:,1] = img2[:,:,0]
    img2[:,:,2] = img2[:,:,0]

    return img2
