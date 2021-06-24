# Color Extraction / Segmentation

import cv2
import numpy as np

img = cv2.imread('Images/A11.jpg')
img = cv2.resize(img, (1000, 500), cv2.INTER_CUBIC)
cv2.imshow('Img', img)

# for red and associates
lower_red = np.array([0, 0, 100], np.uint8)
upper_red = np.array([75, 165, 255], np.uint8)

mask_red = cv2.inRange(img, lower_red, upper_red)
red = cv2.bitwise_and(img, img, mask = mask_red)

# for white background

# cv2.imshow('red', red)

lower_black = np.array([0, 0, 0], np.uint8)
upper_black = np.array([20, 20, 20], np.uint8)

# mask of the background
mask_black = cv2.inRange(red, lower_black, upper_black)
# cv2.imshow('mask_bg', mask_black)

# stacking up the images 3 times
tr = cv2.merge([mask_black, mask_black, mask_black])
# cv2.imshow('tr', tr)
 
# bitwise_or
final = cv2.bitwise_or(red, tr)
# cv2.imshow('final', final)


# for blue and associates
# value by trial and error
lower_blue = np.array([40, 0, 0], np.uint8)
upper_blue = np.array([255, 185, 90], np.uint8)

mask_blue = cv2.inRange(img, lower_blue, upper_blue)
blue = cv2.bitwise_and(img, img, mask = mask_blue)

# cv2.imshow('blue', blue)

black_lower = np.array([0, 0, 0], np.uint8)
black_upper = np.array([20, 20, 20], np.uint8)

mask_black = cv2.inRange(blue, black_lower, black_upper)

tr = cv2.merge([mask_black, mask_black, mask_black])

final = cv2.bitwise_or(blue, tr)
# cv2.imshow('final', final)


# for green and associates
lower_green = np.array([0, 100, 0], np.uint8)
upper_green = np.array([80, 255, 180], np.uint8)

mask_green = cv2.inRange(img, lower_green, upper_green)
green = cv2.bitwise_and(img, img, mask = mask_green)

cv2.imshow('green', green)

lower_black = np.array([0, 0, 0], np.uint8)
upper_black = np.array([20, 20, 20], np.uint8)

mask_black = cv2.inRange(green, lower_black, upper_black)
# cv2.imshow('mask', mask_black)

tr = cv2.merge([mask_black, mask_black, mask_black])
# cv2.imshow('tr', tr)
final = cv2.bitwise_or(tr, green)

cv2.imshow('final', final)
cv2.imwrite('Images/green.jpg', final)

cv2.waitKey(0)
cv2.destroyAllWindows()