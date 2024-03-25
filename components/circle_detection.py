import cv2
import numpy as np
import glob
import os


def detection(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=40, param2=20, minRadius=80, maxRadius=90)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle = circles[0][0]  
        center = (circle[0], circle[1])
        radius = circle[2]

        x = max(center[0] - radius, 0)
        y = max(center[1] - radius, 0)
        width = min(radius * 2, image.shape[1] - x)
        height = min(radius * 2, image.shape[0] - y)

        cv2.circle(image, center, radius, (0, 255, 0), 4)
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Gambar bounding box

    return image



