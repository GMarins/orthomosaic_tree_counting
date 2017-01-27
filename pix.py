
'''Code by Gabriel Marins da Costa

'''
import sys
import time
import numpy as np
from math import sqrt
import cv2
import matplotlib.pyplot as plt


def convolutionFilter(img):
    '''Convolves an image with the kernel. From the documentation:
    Operation is like this: keep the kernel above a pixel add all the 25 pixels below this kernel,
    take its average and eplace the central pixel with the new average value. '''
    kernel = np.ones((5,5),np.float32)/25
    dst=cv2.filter2D(img,-1,kernel)
    cv2.imwrite('convolutionFilterOutput.tif',dst)
    return dst

def bilateralFilter(img):
    '''Bilateral Filter can reduce unwanted noise very well
    while keeping edges fairly sharp. However, it is very slow compared to most filters.
    '''
    w,h=img.shape[::-1]
    diag = sqrt(w**2+h**2)
    #WARNING: Slight changes to the last argument below may cause severe increment in execution time
    blur = cv2.bilateralFilter(img,-1,0.02*diag,9)
    cv2.imwrite('bilateralFilterOutput.tif',blur)
    return blur

def generateHistogram(img):
    '''Generates the contrast histogram of an image'''
    plt.hist(img.ravel(),256,[0,256])
    plt.savefig('histogram.tif')

def thresholdPic(img):
    highThresh, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("thresholded.tif",thresh_img)
    return thresh_img

def equalizeContrast(img):
    '''Increases the contrast of the image based on the histogram using adaptive histogram equalization'''
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    cl = clahe.apply(img)
    cv2.imwrite("claheEqualization.tif",cl)
    return cl

def detectCircles(img_input):
    '''Use the Hough transform to match circles in the given image'''
    img = cv2.medianBlur(img_input,7)
    colored_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    #WARNING: Slight changes to the parameters may cause a severe increment in execution time
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,18,param1=1,param2=18,minRadius=10,maxRadius=17)
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(colored_img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(colored_img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imwrite('circlesMatched.tif',colored_img)
    print("Occurences found: %d" %(len(circles[0])))
    #This result seems to be quite good since the errors kind of compensate the non-detected trees


def norm(p1,p2):
    '''Computes the square of the distance between 2 points'''
    return (p1[1]-p2[1])**2 + (p1[0]-p2[0])**2



def templateMatchingMethod(img_input,template):
    img = cv2.medianBlur(img_input,7)
    colored_img = cv2.cvtColor(img_input,cv2.COLOR_GRAY2BGR)

    w,h = template.shape[::-1]

    result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) #Outputs a np.ndarray

    threshold = 0.7

    loc = np.where(result >= threshold) #Outputs a tuple of np.ndarrays
    loc_filtered = filterMatches(loc, option = 2) #WARNING: option 2 may take much longer to execute

    for pt in loc_filtered: #* operator unzip the tuples
        cv2.rectangle(colored_img,pt,(pt[0] + w, pt[1] + h), (0,255,0), 3)
    cv2.imwrite('templatesMatched.tif',colored_img)
    print("Occurences found: %d" %(len(loc_filtered)))



def filterMatches(matchingResults, option = 2):

    ''' Filters the result from the OpenCV 'match template' function to prevent multiple matches of the same
        tree .
        Given an array of tuples that represent xy coordinates,
        remove from the array the tuples that are sufficiently close
        to the others.

        Keyword argument:
        option -- (default 2)
        If option is 0, returns the input formatted as an array of tuples

        If option is 1 the function checks if the point at position i is close to the point at i+1. If it is,
        remove the point at i+1 from the array. If not, check i+1 and i+2, and so on. Although this method is O(n),
        it is not as accurate as the  option 2, since OpenCV may return close points at different positions in the array.
        Hence, this option return the array with a step of 8 as an average of repeated points per tree obtained after
        some tests (it needs more studies)

        If option is 2 the function will check each point with respect to 800 positions forward.
        Although this method is slower (O(n^2)), the algorithm reduces the size of
        the array during the execution and it is much more accurate the option 1. For the
        example given their execution time are almost the same

        Option 3 is obsolete, rudimental and non-optimized. Return the same result as option 2 but with a much longer execution time
        (~5x)

        '''
    matchings = list(matchingResults[::-1])
    matchingPoints = zip(*matchings)
    matchingPoints = list(matchingPoints)

    if option == 0:
        return matchingPoints

    if option == 1:
        i=0
        while i < len(matchingPoints) - 1: #Since the array is sorted, O(n)
            if norm(matchingPoints[i],matchingPoints[i+1]) < 1000 :
                matchingPoints.pop(i+1)
            else:
                i += 1

        return matchingPoints[::8]

    elif option == 2:
        i=0
        matchingPoints.sort()
        while i < len(matchingPoints) - 1: #Since the array is sorted, O(n)
            j = i+1
            while j < i + 900 and j != len(matchingPoints): #Recomend a step of ~2% of the size of the array
                if norm(matchingPoints[i],matchingPoints[j]) < 1000: #Average squared distance between two adjacent trees
                    matchingPoints.pop(j)
                else:
                    j += 1
            i += 1
        return matchingPoints

    elif option == 3:
        #OBSOLETE
        i=0
        j=1
        while i < len(matchingPoints)-1:
            while j < len(matchingPoints):
                if matchingPoints[i] != matchingPoints[j] and norm(matchingPoints[i],matchingPoints[j]) < 1000:
                    matchingPoints.pop(j)
                else:
                    j +=1
            i += 1
            j = i
        return matchingPoints

    else:
        return None

def main():
    img = cv2.imread(sys.argv[1])
    template = cv2.imread(sys.argv[2],0)
    # img = cv2.imread(input("Enter the name of the image file: "))
    # template = cv2.imread(input('Enter the name of the template file: '),0)

    startTime = time.time()
    print("Process Initiated")

    img_grey=cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    cv2.imwrite('desafio_grey.tif',img_grey)

    print("\nGenerating histogram...")
    generateHistogram(img)

    print("\nEqualizing contrast...")
    img_equalized = equalizeContrast(img_grey)

    print('\nFilterting...\n\n\n')
    convoluted_img = convolutionFilter(img_grey)
    bilateral_filtered_img = bilateralFilter(img_grey)
    thresh_img = thresholdPic(img_grey)


    print('-----------------------------------')
    print("Matching circles")
    detectCircles(bilateral_filtered_img)
    print("Generating output...")

    print('-----------------------------------')
    print("Matching templates")
    try:
        templateMatchingMethod(img_grey,template)
    except:
        print('Matching templates failed, filter option unavailable')
    print("Generating output...")
    print('-----------------------------------')

    print("\nProcess finalized in %.2f seconds" %(time.time()-startTime))
    exit = input('Press ENTER to exit')



if __name__ == '__main__':
    main()
