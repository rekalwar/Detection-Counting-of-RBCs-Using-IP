import cv2
import numpy as np

def detected_cell_counting(image):
    # Convert it to HSV Color Map
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Set the thresholds for RBC Detection
    
    rlower = np.array([160, 60, 100])
    rupper = np.array([180, 200, 255])

    # Find the colors within the specified boundaries and apply the mask
    rmask = cv2.inRange(img, rlower, rupper)

    # Find the contours
    rcontours, hierarchy = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Count the number of RBC's
    rbc_count = 0
    rbc = []
    multiple_cells = []
    for contour in rcontours:
        a = cv2.contourArea(contour) 
        if a > 75 and a <= 300:
            rbc.append(contour)
            rbc_count += 1
        elif a > 300:
            multiple_cells.append(contour)
            if a <= 600:
                rbc_count += 2
            else:
                rbc_count += 3

    # Print the count 
    print(f'Approximate number of RBC: {rbc_count}')

    # Segment the RBC's
    cv2.drawContours(image, multiple_cells, -1, (0, 0, 255), 1)
    cv2.drawContours(image, rbc, -1, (0, 0, 255), 1)

    # Set the thresholds for WBC Detection
    wlower = np.array([120, 50, 20])
    wupper = np.array([140, 255, 255])

    # Find the colors within the specified boundaries and apply the mask
    wmask = cv2.inRange(img, wlower, wupper)

    # Find the contours
    wcontours, hierarchy = cv2.findContours(wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Count and segment the WBC's
    cv2.drawContours(image, wcontours, -1, (255, 0, 0), 1)
    wbc_count = len(wcontours)

    # Print the count
    print(f'Approximate number of WBC: {wbc_count}')
    print(f'Total number of Cells: {rbc_count + wbc_count}')

    # Save the output image
    #cv2.imwrite('static/images/after_count_cell.jpg', image)

    return rbc_count, wbc_count, image