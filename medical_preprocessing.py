import cv2
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/masks/mask_73.png"

image = cv2.imread(path)

px = image[254, 370] #yellow = [36 231 253]
px2 = image[0, 0] #purple = [84 1 68]
print(px)
print(px2)

# cv2.imshow('Window name', image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def isYellow(pixel):
    if pixel[0] == 36:
        return True

def allYellow(image):
    allcoords = []
    for i in range(512):
        for j in range(512):
            pixel = image[i, j]
            if (isYellow(pixel)):
                allcoords.append((i, j))
    return allcoords

print(allYellow(image))