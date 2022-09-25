import cv2
import numpy as np
import os
import pytesseract
import matplotlib.pyplot as plt

def showImage(name,image):
    scale_percent = 50  # percent of original size
    width = 750
    height = 750
    dim = (width, height)

    imgplot = plt.imshow(image)

    plt.show()
    # cv2.waitKey(0)



def table_detection(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # showImage("img_gray", img_gray)

    kernelSizes = [(3, 3), (5, 5), (7, 7)]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

    # img_gray = cv2.dilate(img_gray.copy(), None, iterations= 2)

    (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)
    # showImage("horizontal_lines_img", img_bin)
    kernel_length_v = (np.array(img_gray).shape[1])//120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)
    # showImage("vertical_lines_img", vertical_lines_img)
    kernel_length_h = (np.array(img_gray).shape[1])//120
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)
    # showImage("horizontal_lines_img", horizontal_lines_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    id_list=[]
    name_list=[]
    attendence_list=[]

    if True:
	    cv2.imwrite("./results/table_detect/table_detect__" + img_path.split('/')[-1], table_segment)
	    cv2.imwrite("./results/bb/bb__" + img_path.split('/')[-1], img)


{table_detection('./forms/' + i) for i in os.listdir('./forms/') if i.endswith('.jpeg') or i.endswith('.PNG') or i.endswith('.jpg') or i.endswith('.JPG')}
