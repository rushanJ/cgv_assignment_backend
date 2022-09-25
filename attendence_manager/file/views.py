import os

from django.http import HttpResponse
from rest_framework import parsers

from .models import File
from .serializers import FileSerializer
from rest_framework import generics
from rest_framework import mixins
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters
import requests
import cv2
import numpy as np
import os
import pytesseract
import matplotlib.pyplot as plt
import re


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
    kernel_length_v = (np.array(img_gray).shape[1]) // 120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)
    # showImage("vertical_lines_img", vertical_lines_img)
    kernel_length_h = (np.array(img_gray).shape[1]) // 120
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
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 80 and (h > 20 and h < 500)) and w > 3 * h:
            count += 1
            cropped = img[y:y + h, x:x + w]
            # imgplot = plt.imshow(cropped)
            # plt.show()
            if pytesseract.image_to_string(cropped).strip() == "No" or pytesseract.image_to_string(
                    cropped).strip() == "‘‘Title | ":
                break
            if count % 3 == 1:
                print("id : ")
                id=removeSpecialChars("id",pytesseract.image_to_string(cropped))
                print(pytesseract.image_to_string(cropped))
            if count % 3 == 2:
                print("Name : ")
                name=removeSpecialChars("name",pytesseract.image_to_string(cropped))
            if count % 3 == 0:
                print("Signature : ")

                print(pytesseract.image_to_string(cropped))
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                (tsh, bin) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                bin = cv2.bitwise_not(bin)
                pixelCount = 0
                count255 = 0
                for i in bin:
                    for j in i:
                        pixelCount += 1
                        if j == 255:
                            count255 += 1

                signatureAvailability = (count255 / pixelCount) * 100
                if signatureAvailability>9:
                    attendence= True
                else:
                    attendence=False
                print (id)
                obj = {
                    "userId": id,
                    "name": name ,
                    "moduleCode": "1",
                    "moduleName": "1",
                    "date": "1",
                    "attendence": attendence
                }
                r = requests.post('http://localhost:8000/api/attendence/', data=obj)
                print(r.text)
                print(signatureAvailability)

            # cv2.waitKey(0)
            if True:
                cv2.imwrite("./results/cropped/crop_" + str(count) + "__" + img_path.split('/')[-1], cropped)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if True:
        cv2.imwrite("./results/table_detect/table_detect__" + img_path.split('/')[-1], table_segment)
        cv2.imwrite("./results/bb/bb__" + img_path.split('/')[-1], img)


def removeSpecialChars(type,string):

    if type=='name':string = re.sub(r'^.*?\n', '', string)

    string = string.split("\n")
    print(string[0])
    return string[0]


class FileAPIView(generics.GenericAPIView, mixins.ListModelMixin, mixins.CreateModelMixin, mixins.UpdateModelMixin,
                  mixins.DestroyModelMixin, mixins.RetrieveModelMixin):
    serializer_class = FileSerializer
    queryset = File.objects.all()



    def get(self, request, *args, **kwargs):


        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        req= self.create(request, *args, **kwargs)
        table_detection("4.jpeg")

        return req

    def update(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)



    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)



