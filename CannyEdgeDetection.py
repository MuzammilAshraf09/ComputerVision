#bcsf21m009
#Canny Edge Detection 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Kernals Applying 
def applyCustomKernel(image, kernel):
    rows, cols = image.shape
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            result[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel)

    return result


def calculateMagnitude(prewittX, prewittY):
    return np.sqrt(prewittX**2 + prewittY**2)


def nonMaxSuppression(magnitude, prewittX, prewittY):
    rows, cols= magnitude.shape
    output = np.zeros_like(magnitude, dtype=np.int32)
    angle = np.arctan2(prewittY, prewittX) * 180. / np.pi    #calculates the gradient direction angles
    angle[angle < 0] += 180   # ensuring all angles positive

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output


def doubleThreshold(image, lowThresholdRatio, highThresholdRatio):
    highThreshold = image.max() * highThresholdRatio  # high threshhold
    lowThreshold = highThreshold * lowThresholdRatio    # low threshhold 

    rowsCount, columnsCount = image.shape
    outputImage = np.zeros((rowsCount, columnsCount), dtype=np.uint8)

    strong = 255
    weak = 30

    strongI, strongJ = np.where(image >= highThreshold)   # marked strong that are = or above the high threshold 
    weakI, weakJ = np.where((image <= highThreshold) & (image >= lowThreshold))   # marked as weak that are betwen the high and low 

    outputImage[strongI, strongJ] = strong
    outputImage[weakI, weakJ] = weak

    return outputImage 


def hysteresisEdge(image):
    weak = 30
    strong = 255
    rowsCount, columnsCount = image.shape

    for i in range(1, rowsCount - 1):
        for j in range(1, columnsCount - 1):
            if image[i, j] == weak:
                if (image[i, j-1:j+2] == strong).any() or  (image[i+1, j-1:j+2] == strong).any() or (image[i-1, j-1:j+2] == strong).any():
                    image[i, j] = strong   # If any of the neighboring pixels is a strong edge, the current pixel is also considered a strong edge, so its value in the image is set to strong

                else:
                    image[i, j] = 0

    return image


def cannyEdgeDetection(img, lowRatio, highRatio):

    
    prewittX = applyCustomKernel(img, prewittXKernel)
    prewittY = applyCustomKernel(img, prewittYKernel)

    magnitudeImg = calculateMagnitude(prewittX, prewittY)
    suppressedImg = nonMaxSuppression(magnitudeImg, prewittX, prewittY)

    Image = doubleThreshold(suppressedImg, lowRatio, highRatio)
    finalImage = hysteresisEdge(Image)

    return finalImage


# Prewitt kernels
prewittXKernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

prewittYKernel = np.array([[-1, -1, -1],  [0, 0, 0], [1, 1, 1]])


imageName = input("Enter the name of the image you want to process: ")
# Read the image in grayscale
img = cv.imread(imageName, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (200, 290))

img= cv.GaussianBlur(img, (5,5), 0)

highRatio = float(input("Enter high threshold ratio : "))
lowRatio = float(input("Enter low threshold ratio : "))
# Apply Canny edge detection
edgesInImg = cannyEdgeDetection(img, lowRatio, highRatio)

# Plot the original image
plt.figure(figsize=(10, 15))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# Plot the Canny edge-detected image
plt.figure(figsize=(10, 15))
plt.imshow(cv.cvtColor(edgesInImg, cv.COLOR_BGR2RGB))
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()

# to better show the result..
cv.imshow('win', edgesInImg)
cv.waitKey(0)
cv.destroyAllWindows()


