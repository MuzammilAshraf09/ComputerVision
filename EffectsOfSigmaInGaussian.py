# bcsf21m009
#  Write driver code to compare the effect of sigma (in Gaussian smoothing).

import cv2
import numpy as np
import matplotlib.pyplot as plt



def createGaussianKernel(kernelSize, sigma):
    k=np.fromfunction(lambda x,y: x**2+y**2, (3,3))
    print (k)
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - kernelSize // 2) ** 2 + (y - kernelSize // 2) ** 2) / (2 * sigma ** 2)), (kernelSize, kernelSize))
    normal = kernel / np.sum(kernel)
    return normal


def applyKernelToImage(image, kernel):
    numRows, numCols = image.shape
    resultImage = np.zeros_like(image, dtype=np.float32)
    padding = kernel.shape[0] // 2  # ensures the kernal apply on all elements

    for i in range(padding, numRows-padding):
        for j in range(padding, numCols-padding):
            resultImage[i, j] = np.sum(   image[i-padding:i+padding+1, j-padding:j+padding+1] * kernel)
    resultImage = np.clip(resultImage, 0, 255).astype(   np.uint8)  # make sure the pixels in between 0 to 255
    return resultImage


imageName = input("Enter the name of the image you want to process: ")
# Read the image in grayscale
image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
resizedImage = cv2.resize(image, (200, 290))

# Ask the user to input the sigma values
sigmaValues = []
while True:
    sigmaInput = input("Enter a sigma value or PRESS 'e' to EXIT: ")
    if sigmaInput.lower() == 'e':
        break
    sigmaValues.append(float(sigmaInput))



for sigma in sigmaValues:
    # Create a Gaussian kernel
    gaussianKernel = createGaussianKernel(3, sigma)

    # Apply the kernel to the image
    smoothedImage = applyKernelToImage(resizedImage, gaussianKernel)
    plt.figure(figsize=(10,10))  # Set the figure size

    # Plot the image
    plt.imshow(cv2.cvtColor(smoothedImage, cv2.COLOR_BGR2RGB))
    plt.title(f'Sigma is {sigma}')
    plt.axis('off')
    plt.show()

