# bcsf21m009
# Translation with custom function
import cv2 as cv
import numpy as np


def pointSelectionForShape(event, x, y, flags, param):

    if event == cv.EVENT_LBUTTONDOWN:
        if len(selectedPoints) < noOfPoints:
            print(f"Selected point by the user: ({x}, {y})")
            selectedPoints.append((x, y))
            cv.circle(img, (x, y), radius=3, color=(0, 255, 255), thickness=2)
            cv.imshow("image", img)

        if len(selectedPoints) == noOfPoints:
            # showing the figure of the selected ponits by the user
            cv.polylines(img, [np.array(selectedPoints)], isClosed=True, color=(0, 255, 255), thickness=3)
            # Translating the points from function
            translatedPoints = translationOfPoints(selectedPoints, tx, ty)
            # array of the transled points
            translatedPolygon = np.array(translatedPoints, dtype=np.int32)

            # Drawig  the translated polygon
            cv.polylines(img, [translatedPolygon],isClosed=True, color=(255, 0, 255), thickness=3)
            cv.imshow("image", img)
# function for the transaltion


def translationOfPoints(points, tx, ty):
    translatedPoints = []
    for point in points:
        x, y = point
        translatedPoints.append((x + tx, y + ty))
    return translatedPoints


selectedPoints = []

imgName = input("Enter the name of the image you want to open: ")
# taking the points for translation
tx = float(input("Enter the translation point for x: "))
ty = float(input("Enter the translation point for y: "))

img = cv.imread(imgName)

print("Click on the image to select points for the irregular closed shape polygon")
noOfPoints = int( input("Enter the number of points you want to select (must be more than 2): "))
# validation
while noOfPoints <= 2:
    noOfPoints = int(input("Enter the number of points (must be more than 2): "))

cv.imshow("image", img)
cv.setMouseCallback("image", pointSelectionForShape)
cv.waitKey()
cv.destroyAllWindows()
