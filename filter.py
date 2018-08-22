import cv2
import numpy as np
import functions as f
import morphology as m
import math
import glob

'''
Stage 1 - Noise accounting and marking

Accounting and marking of holes of the depth map after applying a closing lambda
1.

Inputs:
    color
    depth
    depth_closing - Depth map with closing lambda 1

Output:
    holes_marked
    red_holes

Used functions: rgb2gray, threshold, negative, conteo

'''

# Ask user for folder
directory_name = raw_input("-> Directory name: ")

# Read depth images from database folder
images_mcbr = glob.glob(directory_name + '/*MCbR.png')
print images_mcbr

# Read depth images from database folder
images_closing = glob.glob(directory_name + '/*closing.png')
print images_closing

# Read depth images from database folder
images_color = glob.glob(directory_name + '/*.ppm')
print images_color

# Read depth images from database folder
images_depth = glob.glob(directory_name + '/*.pgm')
print images_depth

for fname in images:
    # Inputs
    color = cv2.imread("Image5/view1_2_1.png")
    depth = cv2.imread("Image5/disp1.png")
    depth_mcbr = cv2.imread("Image5/mcbr.png")
    depth_closing = cv2.imread("Image5/closing1.png")

    # Image closing lambda 1 to gray
    depth_closing_gray = f.rgb2gray(depth_closing)
    height, width, channels = depth_closing_gray.shape

    # Threshold to isolate holes
    holes_black = f.threshold(depth_closing_gray, 0, 255)
    # Negative
    holes_white = f.negative(holes_black)

    # Adding mark of zeros around image
    black_image = np.zeros((height + 2, width + 2, 1))
    for j in range(0, height):
        for i in range(0, width):
            black_image[j + 1, i + 1, 0] = holes_white[j, i, 0]

    # Accounting of holes
    holes_marked, objects = f.count(black_image)

    # Marking the holes in red
    red_holes = np.copy(holes_white)
    for j in range(0, height):
        for i in range(0, width):
            if (holes_white[j, i, 0] > 0):
                red_holes[j, i, 0] = 0
                red_holes[j, i, 1] = 0
                red_holes[j, i, 2] = 255

    cv2.imwrite("Test/holes_marked.png", holes_marked)
    cv2.imwrite("Test/holes_red.png", red_holes)

    '''

    Stage 2 - Evaluation

    Inputs:
        depth_mcbr
        holes_marked
        red_holes

    Internals:
        EG - external gradient
        IG - internal gradient

    Outputs:
        holes_gray
    '''

    EG = np.copy(holes_marked)
    IG = np.copy(holes_marked)
    hole_n = np.copy(holes_marked)
    holes_gray = np.copy(depth_mcbr)

    lista = []

    for n in range(1, objects + 1):
        acc1 = 0
        acc2 = 0

        k = 0
        l = 0

        hole_n = f.threshold(holes_marked, n - 1, n + 1)
        EG = m.gradienteExterno(hole_n, 1)
        IG = m.gradienteInterno(hole_n, 1)

        for j in range(0, height):
            for i in range(0, width):
                if EG[j + 1, i + 1, 0] == 255:
                    acc1 = acc1 + depth_mcbr[j, i, 0]
                    k += 1
                if IG[j + 1, i + 1, 0] == 255:
                    acc2 = acc2 + depth_mcbr[j, i, 0]
                    l += 1

        if abs(acc1 / k - acc2 / l) <= 1:
            lista.append(n)

        n += 1

    print lista

    for l in range(0, len(lista)):
        m = lista.pop()
        for j in range(0, height):
            for i in range(0, width):
                if holes_marked[j, i, 0] == m:
                    holes_gray[j, i, 2] = 255

    cv2.imwrite("Test/holes_gray.png", holes_gray)

    '''

    Stage 3 - Isolation

    Inputs:
        holes_marked

    Internals:
        binary_holes
        color_black_border

    Outputs:
        hole_n
        hole_color
        template

    '''

    binary_holes = np.copy(holes_marked)

    for j in range(0, height):
        for i in range(0, width):
            if holes_gray[j, i, 2] == 255:
                binary_holes[j, i, 0] = 255
            else:
                binary_holes[j, i, 0] = 0

    depth_marked, objects = f.count(binary_holes)
    cv2.imwrite("Test/Binary.png", binary_holes)

    depth_black_border = np.zeros((height + 2, width + 2, 3))
    color_black_border = np.zeros((height + 2, width + 2, 3))
    for j in range(0, height):
        for i in range(0, width):
            depth_black_border[j + 1, i + 1, 0] = depth[j, i, 0]
            depth_black_border[j + 1, i + 1, 1] = depth[j, i, 1]
            depth_black_border[j + 1, i + 1, 2] = depth[j, i, 2]
            color_black_border[j + 1, i + 1, 0] = color[j, i, 0]
            color_black_border[j + 1, i + 1, 1] = color[j, i, 1]
            color_black_border[j + 1, i + 1, 2] = color[j, i, 2]

    hole_depth = np.copy(depth_black_border)
    hole_color = np.copy(color_black_border)

    for n in range(1, objects + 1):
        hole_n = f.threshold(depth_marked, n - 1, n + 1)
        cv2.imwrite("Test/" + str(n) + ".png", hole_n)

        for j in range(0, height + 2):
            for i in range(0, width + 2):
                if hole_n[j, i, 0] == 255:
                    hole_depth[j, i, 0] = depth_black_border[j, i, 0]
                    hole_depth[j, i, 1] = depth_black_border[j, i, 1]
                    hole_depth[j, i, 2] = depth_black_border[j, i, 2]
                    hole_color[j, i, 0] = color_black_border[j, i, 0]
                    hole_color[j, i, 1] = color_black_border[j, i, 1]
                    hole_color[j, i, 2] = color_black_border[j, i, 2]
                else:
                    hole_depth[j, i, 0] = 255
                    hole_depth[j, i, 1] = 255
                    hole_depth[j, i, 2] = 255
                    hole_color[j, i, 0] = 255
                    hole_color[j, i, 1] = 255
                    hole_color[j, i, 2] = 255

        cv2.imwrite("Test/" + str(n) + "D.png", hole_depth)
        cv2.imwrite("Test/" + str(n) + "C.png", hole_color)

        min_x = 100000
        max_x = 0

        min_y = 100000
        max_y = 0

        for j in range(0, height + 2):
            for i in range(0, width + 2):
                if hole_depth[j, i, 0] == 0 or hole_depth[j, i, 1] == 0 or hole_depth[j, i, 2] == 0:
                    if i > max_x:
                        max_x = i
                    if i < min_x:
                        min_x = i
                    if j > max_y:
                        max_y = j
                    if j < min_y:
                        min_y = j

        print "Punto " + str(n)
        print min_x
        print max_x
        print min_y
        print max_y

        templateD = hole_depth[min_y-1:max_y+1, min_x-1:max_x+1]
        templateC = hole_color[min_y-1:max_y+1, min_x-1:max_x+1]
        #templateD = hole_depth[min_y:max_y, min_x:max_x]
        #templateC = hole_color[min_y:max_y, min_x:max_x]
        cv2.imwrite("Test/" + str(n) + "TD.png", templateD)
        cv2.imwrite("Test/" + str(n) + "TC.png", templateC)
        templateD = np.uint8(templateD)
        templateC = np.uint8(templateC)
        w, h, c = templateD.shape

        depthAux = np.copy(depth)
        colorAux = np.copy(color)

        depth[min_y:max_y, min_x:max_x] = 0
        color[min_y:max_y, min_x:max_x, 0] = 255 - color[min_y:max_y, min_x:max_x, 0]
        color[min_y:max_y, min_x:max_x, 1] = 255 - color[min_y:max_y, min_x:max_x, 1]
        color[min_y:max_y, min_x:max_x, 2] = 255 - color[min_y:max_y, min_x:max_x, 2]

        '''
        Stage 4 - Template Matching Depth

        Inputs:

        Internals:

        Outputs:

        '''

        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                   'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        m = 1

        dist = height * width
        maxVal_depth = 0.0
        maxVal_color = 0.0
        s_depth = 0.0
        s_color = 0.0
        for meth in methods:
            img = depth.copy()
            method = eval(meth)
            dist_depth = height * width
            coords_best_depth = (0, 0)
            #coords = (0,0)

            criteria = 0

            res = cv2.matchTemplate(img, templateD, method)
            cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

            if meth == 'cv2.TM_CCORR' or meth == 'cv2.TM_CCOEFF' or meth == 'cv2.TM_CCORR_NORMED' or meth == 'cv2.TM_CCOEFF_NORMED':
                threshold = 0.95
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    dist_point = math.sqrt((pt[0] - min_x) ** 2 + (pt[1] - min_y) ** 2)
                    if dist_point < dist:
                        dist = dist_point
                        coords = (pt[0], pt[1])
                        maxVal_depth = maxVal

            if meth == 'cv2.TM_SQDIFF' or meth == 'cv2.TM_SQDIFF_NORMED':
                threshold = 0.05
                loc = np.where(res <= threshold)
                for pt in zip(*loc[::-1]):
                    dist_point = math.sqrt((pt[0] - min_x) ** 2 + (pt[1] - min_y) ** 2)

                    if dist_point < dist:
                        dist = dist_point
                        coords = (pt[0], pt[1])
                        maxVal_depth = maxVal

            cv2.rectangle(img, coords, (coords[0] + h, coords[1] + w), (0, 0, 255), 1)
            cv2.imwrite('Test/img' + str(n) + str(m) + 'D.png', img)

            # if meth == 'cv2.TM_SQDIFF' or meth == 'cv2.TM_SQDIFF_NORMED':
            # 	cv2.rectangle(img, minLoc, (minLoc[0] + w, minLoc[1] + h), (0, 0, 255), 1)
            # 	cv2.imwrite('Test/img' + str(m) + str(n) + 'D.png', img)
            # if meth == 'cv2.TM_CCORR' or meth == 'cv2.TM_CCOEFF' or meth == 'cv2.TM_CCORR_NORMED' or meth == 'cv2.TM_CCOEFF_NORMED':
            # 	cv2.rectangle(img, maxLoc, (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 1)
            # 	cv2.imwrite('Test/img' + str(m) + str(n) + 'D.png', img)

            m += 1

            if dist < dist_depth:
                dist_depth = dist
                coords_best_depth = coords
        cv2.rectangle(img, coords_best_depth, (coords_best_depth[0] + h, coords_best_depth[1] + w), (0, 0, 255), 1)
        img[min_y-1:max_y+1, min_x-1:max_x+1] = depthAux[min_y-1:max_y+1, min_x-1:max_x+1]
        cv2.imwrite('Test/img' + str(n) + 'DB.png', img)

        if dist_depth > 0:
            s_depth = maxVal_depth / dist_depth

        print "S depth: " + str(s_depth) + "/" + str(dist_depth) + "=" + str(s_depth)

        '''
        Stage 5 - Template Matching Color

        Inputs:

        Internals:

        Outputs:

        '''
        m=1
        dist = height*width
        for meth in methods:
            img = color.copy()
            method = eval(meth)
            dist_color = height*width
            #coords_best_color = (0,0)

            res = cv2.matchTemplate(img, templateC, method)
            cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

            if meth == 'cv2.TM_SQDIFF' or meth == 'cv2.TM_SQDIFF_NORMED':
                threshold = 0.05
                loc = np.where( res <= threshold)
                for pt in zip(*loc[::-1]):
                    dist_point = math.sqrt((pt[0] - min_x)**2 + (pt[1] - min_y)**2)
                    if dist_point < dist:
                        dist = dist_point
                        coords = (pt[0], pt[1])
                        maxVal_color = maxVal

            if meth == 'cv2.TM_CCORR' or meth == 'cv2.TM_CCOEFF' or meth == 'cv2.TM_CCORR_NORMED' or meth == 'cv2.TM_CCOEFF_NORMED':
                threshold = 0.95
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    dist_point = math.sqrt((pt[0] - min_x)**2 + (pt[1] - min_y)**2)
                    if dist_point < dist:
                        dist = dist_point
                        coords = (pt[0], pt[1])
                        maxVal_color = maxVal

            cv2.rectangle(img, coords, (coords[0] + h, coords[1] + w), (0, 0, 255), 1)
            cv2.imwrite('Test/img' + str(n) + str(m) + 'C.png', img)

            # if meth == 'cv2.TM_SQDIFF' or meth == 'cv2.TM_SQDIFF_NORMED':
            # 	cv2.rectangle(img, minLoc, (minLoc[0] + w, minLoc[1] + h), (0, 0, 255), 1)
            # 	cv2.imwrite('Test/img' + str(m) + str(n) + 'D.png', img)
            # if meth == 'cv2.TM_CCORR' or meth == 'cv2.TM_CCOEFF' or meth == 'cv2.TM_CCORR_NORMED' or meth == 'cv2.TM_CCOEFF_NORMED':
            # 	cv2.rectangle(img, maxLoc, (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 1)
            # 	cv2.imwrite('Test/img' + str(m) + str(n) + 'D.png', img)

            m += 1

            if dist < dist_color:
                dist_color = dist
                coords_best_color = coords
        img[min_y-1:max_y+1, min_x-1:max_x+1] = colorAux[min_y-1:max_y+1, min_x-1:max_x+1]
        cv2.rectangle(img, coords_best_color, (coords_best_color[0] + h, coords_best_color[1] + w), (0, 0, 255), 1)
        cv2.imwrite('Test/img' + str(n) + 'CB.png', img)

        if dist_color > 0:
            s_color = maxVal_color / dist_color

        print "S color: " + str(s_color) + "/" + str(dist_color) + "=" + str(s_color)

        '''
        Criteria 1

        Taking the best match and overwrite it over the region that contains the hole

        '''

        # if s_depth > s_color:
        #     depth_mcbr[min_y:max_y, min_x:max_x, 0] = depth_mcbr[coords_best_depth[1]:coords_best_depth[1]+w, coords_best_depth[0]:coords_best_depth[0]+h, 0]
        #     depth_mcbr[:,:,1] = depth_mcbr[:,:,0]
        #     depth_mcbr[:,:,2] = depth_mcbr[:,:,0]
        #     cv2.imwrite("Test/Filtered_Depth" + str(n) + ".png", depth_mcbr)
        # if s_color > s_depth:
        #     depth_mcbr[min_y:max_y, min_x:max_x, 0] = depth_mcbr[coords_best_color[1]:coords_best_color[1]+w, coords_best_color[0]:coords_best_color[0]+h, 0]
        #     depth_mcbr[:,:,1] = depth_mcbr[:,:,0]
        #     depth_mcbr[:,:,2] = depth_mcbr[:,:,0]
        #     cv2.imwrite("Test/Filtered_Color" + str(n) + ".png", depth_mcbr)

        # '''
        # Criteria 2
        #
        # Taking the gray level average of best match and overwrite it over the hole to fix
        #
        # '''
        # acc = []
        # average = 0.0
        # if s_depth > s_color:
        #     for j in range(coords_best_depth[1], coords_best_depth[1]+h):
        #         for i in range(coords_best_depth[0], coords_best_depth[0]+w):
        #             if j < height and i < width:
        #                 acc.append(depth_mcbr[j,i,0])
        # if s_depth < s_color:
        #     for j in range(coords_best_color[1], coords_best_color[1]+h):
        #         for i in range(coords_best_color[0], coords_best_color[0]+w):
        #             if j < height and i < width:
        #                 acc.append(depth_mcbr[j,i,0])
        #
        # average = sum(acc) / len(acc)
        # print average
        #
        # for j in range(min_y, max_y):
        #     for i in range(min_x, max_x):
        #         if holes_gray[j,i,2] == 255:
        #             depth_mcbr[j,i,0] = int(average)
        #
        # depth_mcbr[:,:,1] = depth_mcbr[:,:,0]
        # depth_mcbr[:,:,2] = depth_mcbr[:,:,0]
        #
        # cv2.imwrite("Test/Filtered_average" + str(n) + ".png", depth_mcbr)

        '''
        Criteria 3

        Taking the gray level average of best match and overwrite it over the hole to fix

        '''
        acc = []
        average = 0.0

        if s_depth > s_color:
            print coords_best_depth[0]
            print coords_best_depth[1]
            print coords_best_depth[0]+h
            print coords_best_depth[1]+w
        if s_depth < s_color:
            print coords_best_color[0]
            print coords_best_color[1]
            print coords_best_color[0]+h
            print coords_best_color[1]+w

        if s_depth > s_color:
            for j in range(0, h):
                for i in range(0, w):
                    if coords_best_depth[0]+j < width and coords_best_depth[1]+i < height and min_y + j < height and min_x + i < width:
                        if holes_gray[min_y + j, min_x + i, 2] == 255:
                            acc.append(depth_mcbr[coords_best_depth[1]+j, coords_best_depth[0]+i, 0])
        if s_depth < s_color:
            for j in range(0, h):
                for i in range(0, w):
                    if coords_best_color[0]+j < width and coords_best_color[1]+i < height and min_y + j < height and min_x + i < width:
                        if holes_gray[min_y + j, min_x + i, 2] == 255:
                            acc.append(depth_mcbr[coords_best_color[1]+j, coords_best_color[0]+i, 0])

        if s_depth == 0.0 and s_color == 0.0:
            for j in range(min_y, max_y):
                for i in range(min_x, max_x):
                    acc.append(depth_mcbr[j, i, 0])

        if len(acc) != 0:
            average = sum(acc) / len(acc)
        else:
            average = 0

        print average

        for j in range(min_y, max_y):
            for i in range(min_x, max_x):
                if holes_gray[j,i,2] == 255:
                    depth_mcbr[j,i,0] = int(average)

        depth_mcbr[:,:,1] = depth_mcbr[:,:,0]
        depth_mcbr[:,:,2] = depth_mcbr[:,:,0]
        cv2.imwrite("Test/Filtered_average" + str(n) + ".png", depth_mcbr)
