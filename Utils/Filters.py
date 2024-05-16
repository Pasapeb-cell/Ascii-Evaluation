import cv2
def applyTransforms(image):
    img = cv2.threshold(image,127,255,cv2.THRESH_BINARY)[1]  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,7))
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img