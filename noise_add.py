#!/usr/bin/env python2

import cv2
import numpy
import glob
import random

holes = 5

# Ask user for folder
directory_name = raw_input("-> Directory name: ")

# Read images from database folder
images = glob.glob(directory_name + '/*.pgm')
print images

# Show images
for fname in images:
    image = cv2.imread(fname, 0)
    cv2.imshow("Test", image)
    cv2.waitKey(100)

    h = image.shape[0]
    w = image.shape[1]

    print h, w

    for i in range(0, holes):
        # Generate random coordinates x and y
        x = random.random()
        y = random.random()

        # axis1 = random.random()
        # axis2 = random.random()
        # angle_start = random.random()
        # angle_end = random.random()
        # angle = random.random()

        radius = random.random()

        height = random.random()

        # Add random noise
        # cv2.ellipse(image, (int(x*w),int(y*h)), (int(axis1*w*0.3),int(axis2*h*0.3)), angle*360, angle_start*360, angle_end*360, (0), -1)
        if i%2 == 0:
            cv2.circle(image, (int(x*w),int(y*h)), int(radius*0.1*w), (0), -1)
        else:
            cv2.rectangle(image, (int(x*w),int(y*h)), (int(x*w+height*w*0.2),int(y*h+height*h*0.2)), (0), -1)
        cv2.imshow("Test", image)
        cv2.waitKey(100)

        # Save images
    cv2.imwrite(fname[0:len(fname)-4]+"-noise.png", image)
