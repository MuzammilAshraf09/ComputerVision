# bcsf21m009
# Scaling of the polygon from center of the polygon
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
        cv.polylines(img, [np.array(selectedPoints)],isClosed=True, color=(0, 255, 255), thickness=3)
        # Calculate the centr of the original polygon
        centrX = 0
        centrY = 0
        for point in selectedPoints:
            centrX += point[0]
            centrY += point[1]

        centrX /= len(selectedPoints)
        centrY /= len(selectedPoints)

        # Scale the points around the centr of the polygon
        scaledPoints = scalingOfPoints(selectedPoints, scale, (centrX, centrY))
        scaledPolygon = np.array(scaledPoints, dtype=np.int32)

        cv.polylines(img, [scaledPolygon], isClosed=True,color=(255, 0, 255), thickness=3)
        cv.imshow("image", img)


def scalingOfPoints(points, scaleFactor, center):
    scaledPoints = []
    for point in points:
        x, y = point
        # adjust the scale points relative to center
        scaledX = center[0] + (x - center[0]) * scaleFactor
        scaledY = center[1] + (y - center[1]) * scaleFactor
        scaledPoints.append((scaledX, scaledY))
    return scaledPoints


selectedPoints = []

imgName = input("Enter the name of the image you want to open: ")
scale = float(input("Enter the scaling factor keeping in veiw that 0.5 for half size, 2.0 for double size: "))
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
