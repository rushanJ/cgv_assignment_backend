def invert_area(image, x, y, w, h, display=False):
    ones = np.copy(image)
    ones = 1

    # image[y:y + h, x:x + w] = ones * 255 - image[y:y + h, x:x + w]
    image = ones * 255 - image

    if (display):
        cv.imshow("inverted", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return image


def detect(cropped_frame, config, is_number=False):
    if (is_number):
        text = pytesseract.image_to_string(cropped_frame, config=config)
    else:
        text = pytesseract.image_to_string(cropped_frame, config=config)
    return text


def draw_text(src, x, y, w, h, text):
    cFrame = np.copy(src)
    cv.rectangle(cFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.putText(cFrame, "text: " + text, (50, 50), cv.FONT_HERSHEY_SIMPLEX,
               2, (0, 0, 0), 5, cv.LINE_AA)

    return cFrame


def erode(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erosion = cv.dilate(img, kernel, iterations=2)
    return img_erosion
