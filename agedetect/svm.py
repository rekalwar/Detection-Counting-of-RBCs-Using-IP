'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn import metrics

# Dataset path
dataset_path = 'E://BE Final Year Projects//RBCs Projects//LeukemiaDetection-master//dataset//'

# Class labels
classes = {'healthy': 0, 'leukemia': 1}

# Load images and labels
X = []
Y = []

for cls, label in classes.items():
    class_path = os.path.join(dataset_path, cls)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (200, 200))
        X.append(img)
        Y.append(label)

# Convert to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Reshape images for model compatibility
X_updated = X.reshape(len(X), -1)

# Split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=.20)

# Features scaling
xtrain = xtrain / 255
xtest = xtest / 255

# PCA for feature selection
pca = PCA(.98)
pca_train = pca.fit_transform(xtrain)
pca_test = pca.transform(xtest)

# Train the model
sv = LinearSVC(dual='auto', C=0.1)

sv.fit(pca_train, ytrain)

# Evaluate the model
print("Training Score:", sv.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))

# Make predictions
pred = sv.predict(pca_test)

# Display accuracy
print("Accuracy:", accuracy_score(ytest, pred))

# Display misclassified samples
misclassified_indices = np.where(ytest != pred)
print("Misclassified Samples Indices:", misclassified_indices)

# Display misclassified sample details
for misclassified_index in misclassified_indices[0]:
    print(f"Predicted: {pred[misclassified_index]}, Actual: {ytest[misclassified_index]}")

# Visualize test samples
def visualize_test_samples():
    dec = {0: 'healthy', 1: 'leukemia'}
    plt.figure(figsize=(12, 8))
    test_images_path = os.path.join(dataset_path, 'Test')
    for i, img_file in enumerate(os.listdir(test_images_path)[:20], start=1):
        plt.subplot(4, 5, i)
        img_path = os.path.join(test_images_path, img_file)
        img = cv2.imread(img_path, 0)
        img1 = cv2.resize(img, (200, 200))
        img1 = img1.reshape(1, -1) / 255
        p = sv.predict(pca.transform(img1))
        plt.title(dec[p[0]])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

# Visualize test samples
visualize_test_samples()
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
'''
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
                                      properties=['label', 'area', 'equivalent_diameter', 'mean_intensity', 'solidity'])

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
    df.to_csv('safal.csv',index=False) 

image_path = 'dataset/bloodimg1.jpg'
image = cv2.imread(image_path)
featureExtraction(image)

# Classification/Recognition: SVM
dataread = pd.read_csv('safal.csv')
print(dataread.head())
# Assuming you have a dataset with features and target ('target' being the label)
X = dataread.drop(['target'], axis=1)
y = dataread['target']

# Ensure that the 'target' column is present in your DataFrame

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['ALL', 'MLL'])
print(confusion)

'''























# Classification/Recognition: SVM
dataread = pd.read_csv('leukemia.csv')
print(dataread.head())
# Assuming you have a CSV file with your dataset

# Assuming you have a dataset with features and target ('target' being the label)
X = dataread.drop(['label'], axis=1)
y = dataread['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['ALL', 'MLL'])
print(confusion)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from wrapper import leukemia
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# %matplotlib inline
dataread = pd.read_csv('safal.csv')
print(dataread.head())
# Assuming you have a CSV file with your dataset

# Assuming you have a dataset with features and target ('target' being the label)
X = dataread(['label'], axis=1)
y = dataread['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['ALL', 'MLL'])
print(confusion)
