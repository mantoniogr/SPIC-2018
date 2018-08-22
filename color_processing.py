#!/usr/bin/env python2

import cv2
import numpy
import glob

import color as c

# Ask user for folder
directory_name = raw_input("-> Directory name: ")

# Read depth images from database folder
images_depth = glob.glob(directory_name + '/*.png')
print images_depth

# Read color images from database folder
images_color = glob.glob(directory_name + '/*.ppm')
print images_color

#
for fname in images_color:
    image_color = cv2.imread(fname)
    color_segm = c.color_segmentation(image_color, 'hs')
    false_color = c.color_assignation(image_color, color_segm)
    cv2.imwrite(fname[0:len(fname)-4]+"-hs.png", false_color)

    color_segm = c.color_segmentation(image_color, 'ls')
    false_color = c.color_assignation(image_color, color_segm)
    cv2.imwrite(fname[0:len(fname)-4]+"-ls.png", false_color)
