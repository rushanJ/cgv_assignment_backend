# Importing Python Libraries
import pytesseract
import cv2 as cv
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import PIL
from matplotlib import pyplot as plt

# Importing User Defined Functions
from preprocessing import invert_area, draw_text, detect, get_grayscale, get_binary
from ROI_selection import detect_lines, get_ROI, table_detection

# Defining a configuration for PyTesseract OCR
charConfig = r"--psm 6 --oem 3"
numConfig =  r"--psm 10 --oem 3"

# Reading and saving the source Image
filename = "input/source.jpeg"
src = cv.imread(cv.samples.findFile(filename))

# Resize Image while keeping the aspect ratio
imgDownscale = 50 # Image Downscaling Size
imgWidth = int(src.shape[1] * imgDownscale / 100)
imgHeight = int(src.shape[0] * imgDownscale / 100)
imgDim = (imgWidth, imgHeight)

image = cv.resize(src, imgDim, interpolation = cv.INTER_AREA)

# Grayscaling the images to prepare it for edge detection
grayscale = get_grayscale(image)
cv.imwrite("images/" + "Grayscale" + ".png", grayscale)

# Get Binary Image
binary = cv.GaussianBlur(grayscale, (1, 1), 0)
binary = get_binary(binary)
cv.imwrite("images/" + "Binary" + ".png", binary)

# Canny Edge detection
canny = cv.GaussianBlur(grayscale, (1, 1), 0)
canny = cv.Canny(canny, 400, 410)
cv.imwrite("images/" + "Edges" + ".png", canny)

horizontal, vertical = detect_lines(image, title='Hough Lines', minLinLength=20, threshold=50, maxLineGap=4, display=True, write=True)

#Invert Area
left_line_index = 1
right_line_index = 5
top_line_index = 4
bottom_line_index = 10

cropped_image, (x, y, w, h) = get_ROI(image, horizontal, vertical, left_line_index,right_line_index, top_line_index, bottom_line_index)
cv.imwrite("images/" + "Cropped" + ".png", cropped_image)

# Inverted Binary Image
inverted_binary = invert_area(grayscale, x, y, w, h, display=True)
cv.imwrite("images/" + "Inverted Binary" + ".png", inverted_binary)

output = table_detection(src)
print(output)
