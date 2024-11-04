import cv2
import numpy as np
import math

# img1 is the image with the color tones we want
img1 = cv2.imread("night.jpg")

# img2 is the image whose colors we will manipulate
img2 = cv2.imread("river.jpg")


# calculate average of color intensities
def get_average(image):

    rows = image.shape[0]
    cols = image.shape[1]

    totalPixels = rows*cols

    bAverage = 0
    gAverage = 0
    rAverage = 0

    # iterate over pixels to find sumof each color intensity within photo
    for i in range(rows):
        for j in range(cols):

            bIntensity = image[i,j,0]
            gIntensity = image[i,j,1]
            rIntensity = image[i,j,2]

            bAverage += float(bIntensity)
            gAverage += float(gIntensity)
            rAverage += float(rIntensity)

    # calculate average
    bAverage /= totalPixels
    gAverage /= totalPixels
    rAverage /= totalPixels
    

    

    return bAverage, gAverage, rAverage


# calculate standard deviation of image
def get_standard_dev(image, avg, color):

    rows = image.shape[0]
    cols = image.shape[1]

    varianceSum = 0

    #iterate over pixels to determine sum of variances
    for i in range(rows):
        for j in range(cols):
            difference = image[i,j,color]-avg
            variance = difference**2
            varianceSum += variance

    # compute average of variance
    varianceSum = varianceSum/(rows*cols)

    # compute standard deviation
    standardDev = ((varianceSum)**0.5)

    return standardDev



def generate_image(image):

    
    rows = image.shape[0]
    cols = image.shape[1]

    # create new image
    mappedImage = np.zeros((rows, cols, 3), np.float32)

    # find averages of color intensities
    oldBAvg, oldGAvg, oldRAvg = get_average(img2)
    newBAvg, newGAvg, newRAvg = get_average(img1)

  

    # find standard deviation of colors
    bOldSD = get_standard_dev(img2, oldBAvg, 0)
    gOldSD = get_standard_dev(img2, oldGAvg, 1)
    rOldSD = get_standard_dev(img2, oldRAvg, 2)

    

    bNewSD = get_standard_dev(img1, newBAvg, 0)
    gNewSD = get_standard_dev(img1, newGAvg, 1)
    rNewSD = get_standard_dev(img1, newRAvg, 2)

  

    
    # iterate over pixels to set new values 
    for i in range(rows):
        for j in range(cols):

            newBvalue = (((bNewSD/bOldSD)*(image[i,j,0]-oldBAvg))+newBAvg)
            mappedImage[i,j,0] = newBvalue
            newGvalue = (((gNewSD/gOldSD)*(image[i,j,1]-oldGAvg))+newGAvg)
            mappedImage[i,j,1] = newGvalue
            newRvalue = (((rNewSD/rOldSD)*(image[i,j,2]-oldRAvg))+newRAvg)
            mappedImage[i,j,2] = newRvalue

 

    

    cv2.imshow("Tone Mapped Image", mappedImage/255.0)
    cv2.imwrite("ToneMappedImage.jpg", mappedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



generate_image(img2)
            
            

    
