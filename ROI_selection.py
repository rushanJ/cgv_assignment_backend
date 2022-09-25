import cv2 as cv
import numpy as np
import pytesseract

charConfig = r"--psm 6 --oem 3"

def is_vertical(line):
    return line[0] == line[2]


def is_horizontal(line):
    return line[1] == line[3]

def overlapping_filter(lines, sorting_index):
    filtered_lines = []

    lines = sorted(lines, key=lambda lines: lines[sorting_index])
    separation = 1
    for i in range(len(lines)):
        l_curr = lines[i]
        if (i > 0):
            l_prev = lines[i - 1]
            if ((l_curr[sorting_index] - l_prev[sorting_index]) > separation):
                filtered_lines.append(l_curr)
        else:
            filtered_lines.append(l_curr)

    return filtered_lines


def detect_lines(image, title='default', rho=1, theta=np.pi / 180, threshold=120, minLinLength=10, maxLineGap=6, display=False, write=False):

# image - source image
# rho — distance resolution of the accumulator in pixels
# theta — Angle resolution of the accumulator in radians
# threshold — Accumulator threshold parameter. Only those lines are returned that get enough votes
# line — Output vector of lines. Here is set to None, the value is saved to linesP
# minLineLength — Minimum line length. Line segments shorter than that are rejected
# maxLineGap — Maximum allowed gap between points on the same line to link them

    if image is None:
        print('Error opening image!')
        return -1

    # Resize Image
    imgDownscale = 50
    imgWidth = int(image.shape[1] * imgDownscale / 100)
    imgHeight = int(image.shape[0] * imgDownscale / 100)
    imgDim = (imgWidth, imgHeight)


    # Copy edges to the images that will display the results
    imageCopy = np.copy(image)
    imageCopy = cv.resize(imageCopy, imgDim, interpolation=cv.INTER_AREA)

    # Grayscale the image
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Canny Edge Detection
    canny = cv.resize(grayscale, imgDim, interpolation=cv.INTER_AREA)
    canny = cv.GaussianBlur(canny, (1, 1), 0)
    canny = cv.Canny(canny, 400, 410)

    cv.imshow("Binary", canny)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # linesP = cv.HoughLinesP(dst, 1 , np.pi / 180, 50, None, 290, 6)
    linesP = cv.HoughLinesP(canny, rho, theta, threshold, None, minLinLength, maxLineGap)

    horizontal_lines = []
    vertical_lines = []

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            if is_vertical(l):
                vertical_lines.append(l)

            elif is_horizontal(l):
                horizontal_lines.append(l)

        horizontal_lines = overlapping_filter(horizontal_lines, 1)
        vertical_lines = overlapping_filter(vertical_lines, 0)

    if display:
        for i, line in enumerate(horizontal_lines):
            cv.line(imageCopy, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3, cv.LINE_AA)

            cv.putText(imageCopy, str(i) + "h", (line[0] + 5, line[1]), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1, cv.LINE_AA)

        for i, line in enumerate(vertical_lines):
            cv.line(imageCopy, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv.LINE_AA)
            cv.putText(imageCopy, str(i) + "v", (line[0], line[1] + 5), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.imshow("Source", imageCopy)
        cv.waitKey(0)
        cv.destroyAllWindows()

    if write:
        cv.imwrite("images/" + title + ".png", imageCopy);

    return (horizontal_lines, vertical_lines)

def get_cropped_image(image, x, y, w, h):
    cropped_image = image[ y:y+h , x:x+w ]
    return cropped_image


def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index,offset=4):
    x1 = vertical[left_line_index][2] + offset
    y1 = horizontal[top_line_index][3] + offset
    x2 = vertical[right_line_index][2] - offset
    y2 = horizontal[bottom_line_index][3] - offset

    w = x2 - x1
    h = y2 - y1

    cropped_image = get_cropped_image(image, x1, y1, w, h)

    return cropped_image, (x1, y1, w, h)

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