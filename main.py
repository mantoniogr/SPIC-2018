#!/usr/bin/env python2

import cv2
import numpy
import glob
import random

holes = 3

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

        axis1 = random.random()
        axis2 = random.random()

        angle_start = random.random()
        angle_end = random.random()

        cv2.ellipse(image, (int(x*w),int(y*h)), (int(axis1*w),int(axis2*h)), 45, angle_start, angle_end, (255), 0)
        cv2.imshow("Test", image)
        cv2.waitKey(100)

# Add random noise


# Save images
