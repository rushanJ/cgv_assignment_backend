# Inverted Binary Image
inverted_binary = invert_area(grayscale, x, y, w, h, display=True)
cv.imwrite("images/" + "Inverted Binary" + ".png", inverted_binary)

output = table_detection(src)
print(output)
