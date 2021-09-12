import cv2
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

mask_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/masks/mask_73.png"
image_path = "NSCLC2 - Lung_Cancers3/manifest-1603198545583/NSCLC-Radiomics/LUNG1-001/images/image_73.png"

mask = cv2.imread(mask_path)
image = cv2.imread(image_path)

px = mask[254, 370] #yellow = [36 231 253]
px2 = mask[0, 0] #purple = [84 1 68]
print(px)
print(px2)

# cv2.imshow('Window name', image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def isYellow(pixel):
    ''' Return True if the pixel of the mask is yellow (if it is in the segmented zone) '''
    if pixel[0] == 36:
        return True


def crop(image, y, x): #vertical, horizontal
    ''' Return a 32x32 image with top left corner of coordonate(y, x) '''
    crop_img = image[y:y + 32, x:x + 32]
    cv2.imshow('Cropped', crop_img)
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return crop_img


def isFullYellow(mask, y, x):
    ''' Return True if the 4 corners of the 32x32 image are yellow '''
    if isYellow(mask[y, x]) and isYellow(mask[y + 32, x]) and isYellow(mask[y, x + 32]) and isYellow(mask[y + 32, x + 32]):
        return True


def allFullYellow(mask):
    ''' Return a list of all coordonates of the image that are full yellow '''
    allcoords = []
    for y in range(480):
        for x in range(480):
            if (isFullYellow(mask, y, x)):
                allcoords.append((y, x))
    return allcoords


def isFullPurple(mask, y, x):
    ''' Return True if the 4 corners of the 32x32 image are purple '''
    if not (isYellow(mask[y, x]) or isYellow(mask[y + 32, x]) or isYellow(mask[y, x + 32]) or isYellow(mask[y + 32, x + 32])):
        return True


def allFullPurple(mask):
    ''' Return a list of all coordonates of the image that are full purple '''
    allcoords = []
    for y in range(480):
        for x in range(480):
            if (isFullPurple(mask, y, x)):
                allcoords.append((y, x))
    return allcoords


fullyellows = allFullYellow(mask)

crop(image, fullyellows[200][0], fullyellows[200][1])
