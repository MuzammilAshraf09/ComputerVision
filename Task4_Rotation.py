#bcsf21m009
# Rotation of the  neWpolygon referece to the center of the original polygon 

import cv2 as cv
import numpy as np
import math


def pointSelectionForShape(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(selectedPoints) < noOfPoints:
            print(f"Selected point by the user: ({x}, {y})")
            selectedPoints.append((x, y))
            cv.circle(img, (x, y), radius=3, color=(0, 255, 255), thickness=2)
            cv.imshow("image", img)

    if len(selectedPoints) == noOfPoints:
        cv.polylines(img, [np.array(selectedPoints)],isClosed=True, color=(0, 255, 255), thickness=3)

        # Calculate the centr of the original polygon
        centrX = 0
        centrY = 0
        for point in selectedPoints:
            centrX += point[0]
            centrY += point[1]

        centrX /= len(selectedPoints)
        centrY /= len(selectedPoints)

        # Rotate the points around the centroid
        rotatedPoints = rotationOfPoints(selectedPoints, angle, (centrX, centrY))
        rotatedPolygon = np.array(rotatedPoints, dtype=np.int32)

        cv.polylines(img, [rotatedPolygon], isClosed=True,color=(255, 0, 255), thickness=3)
        cv.imshow("image", img)


def rotationOfPoints(points, angle, center):
    rotatedPoints = []
    for point in points:
        x, y = point
        # Convert angle to radians
        radAngle= math.radians(angle)
        # first by subtractinf the center coordinates from point coordiantes shift the center to 0,0 then rotate and shifr back to orignal center by adding center coordinates 
        rotatedX = center[0] + (x - center[0]) * math.cos(radAngle) - (y - center[1]) * math.sin(radAngle)
        rotatedY = center[1] + (x - center[0]) * (-math.sin(radAngle)) + (y - center[1]) * math.cos(radAngle)
        rotatedPoints.append((rotatedX, rotatedY))
    return rotatedPoints


selectedPoints = []

imgName = input("Enter the name of the image you want to open: ")
angle = float(input("Enter the rotation angle in degrees: "))
img = cv.imread(imgName)

print("Click on the image to select points for the irregular closed shape polygon")
noOfPoints = int(input("Enter the number of points you want to select (must be more than 2): "))
# Validation
while noOfPoints <= 2:
    noOfPoints = int(input("Enter the number of points (must be more than 2): "))

cv.imshow("image", img)
cv.setMouseCallback("image", pointSelectionForShape)
cv.waitKey()
cv.destroyAllWindows()
