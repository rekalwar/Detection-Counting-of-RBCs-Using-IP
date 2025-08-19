import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, color

def featureExtraction(img):
    cells = img[:, :, 0]
    pixels_to_um = 0.454
    
    ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    from skimage.segmentation import clear_border
    opening = clear_border(opening)  # Remove edge-touching grains
    plt.imshow(opening, cmap='gray')  # This is our image to be segmented further using watershed
    
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    plt.imshow(sure_bg, cmap='gray')  # Dark region is our sure background

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    plt.imshow(dist_transform, cmap='gray')  # Dist transformed img.
    print(dist_transform.max())  # gives about 21.9
    
    ret2, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    plt.imshow(sure_fg, cmap='gray')
    sure_fg = np.uint8(sure_fg)  # Convert to uint8 from float
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    plt.imshow(unknown, cmap='gray')
    
    ret3, markers = cv2.connectedComponents(sure_fg)
    plt.imshow(markers)
    markers = markers + 10

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    plt.imshow(markers, cmap='jet')  # Look at the 3 distinct regions.

    # Now we are ready for watershed filling.
    markers = cv2.watershed(img, markers)

    # Let us color boundaries in yellow.
    # Remember that watershed assigns boundaries a value of -1
    img[markers == -1] = [0, 255, 255]

    # label2rgb - Return an RGB image where color-coded labels are painted over the image.
    img2 = color.label2rgb(markers, bg_label=0)
    imr1 = cv2.resize(img, (960, 540))
    imr2 = cv2.resize(img2, (960, 540))
    plt.imshow(img2)

    # cv2.imshow('Overlay on original image', imr1)
    # cv2.imshow('Colored Grains', imr2)
    cv2.waitKey(0)

    ########################################################################
    # Now, time to extract properties of detected cells
    # Directly capturing props to pandas dataframe

    props = measure.regionprops_table(markers, cells,
                                      properties=['label', 'area', 'equivalent_diameter', 'mean_intensity', 'solidity', 'orientation', 'perimeter'])
    
    # Load into dataset:
    df = pd.DataFrame(props)
    print(df.head())

    # To delete small regions...
    df = df[df['area'] > 50]
    print(df.head())

    #######################################################
    # Convert to micron scale
    df['area_sq_microns'] = df['area'] * (pixels_to_um ** 2)
    df['equivalent_diameter_microns'] = df['equivalent_diameter'] * pixels_to_um
    print(df.head())
    df.to_csv('safal.csv')

# Assuming you have an image file to pass to the function
# img = cv2.imread('dataset/leukemia/infected cell.jpg')
# featureExtraction(img)