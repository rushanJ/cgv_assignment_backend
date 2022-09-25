# Canny Edge detection
canny = cv.GaussianBlur(grayscale, (1, 1), 0)
canny = cv.Canny(canny, 400, 410)
cv.imwrite("images/" + "Edges" + ".png", canny)

horizontal, vertical = detect_lines(image, title='Hough Lines', minLinLength=20, threshold=50, maxLineGap=4, display=True, write=True)

# Inverted Binary Image
inverted_binary = invert_area(grayscale, x, y, w, h, display=True)
cv.imwrite("images/" + "Inverted Binary" + ".png", inverted_binary)

output = table_detection(src)
print(output)
