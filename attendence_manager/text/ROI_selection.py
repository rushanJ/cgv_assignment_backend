def table_detection(image):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    kernelSizes = [(3, 3), (5, 5), (7, 7)]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    img_gray = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, kernel)

    # img_gray = cv.dilate(img_gray.copy(), None, iterations= 2)
    (thresh, img_bin) = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    img_bin = cv.bitwise_not(img_bin)

    kernel_length_v = (np.array(img_gray).shape[1]) // 120
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv.dilate(im_temp1, vertical_kernel, iterations=3)

    # showImage("vertical_lines_img", vertical_lines_img)
    kernel_length_h = (np.array(img_gray).shape[1]) // 120
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv.dilate(im_temp2, horizontal_kernel, iterations=3)

    # showImage("horizontal_lines_img", horizontal_lines_img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    table_segment = cv.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv.erode(cv.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv.threshold(table_segment, 0, 255, cv.THRESH_OTSU)

    # Keywords
    keywords = ['student_no', 'student_name', 'signature']

    dict_attendence = {}
    for keyword in keywords:
        dict_attendence[keyword] = []

    contours, hierarchy = cv.findContours(table_segment, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if (w > 80 and (h > 20 and h < 500)) and w > 3 * h and count < 18:
            count += 1
            cropped = image[y:y + h, x:x + w]
            cv.imwrite("cropped/" + str(count) + ".png", cropped);
            if pytesseract.image_to_string(cropped).strip() == "No" or pytesseract.image_to_string(
                    cropped).strip() == "‘‘Title | ":
                break
            if count % 3 == 0:
                id = pytesseract.image_to_string(cropped)
                dict_attendence['student_no'].append(id)
            if count % 3 == 2:
                name = pytesseract.image_to_string(cropped)
                dict_attendence['student_name'].append(name)
            if count % 3 == 1:
                gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
                (tsh, bin) = cv.threshold(gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                bin = cv.bitwise_not(bin)
                pixelCount = 0
                count255 = 0
                for i in bin:
                    for j in i:
                        pixelCount += 1
                        if j == 255:
                            count255 += 1

                signatureAvailability = (count255 / pixelCount) * 100
                if signatureAvailability > 7.0:
                    availability = 'Present'
                    dict_attendence['signature'].append(availability)
                else:
                    availability = 'Absent'
                    dict_attendence['signature'].append(availability)
    return dict_attendence