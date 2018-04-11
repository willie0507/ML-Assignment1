from skimage import io, color
import numpy as np
from os import walk
from os.path import join

train_data = []  # (filename, nd-array form, label)
test_data = []  # (filename, nd-array form, (SAD label, SSD label))

for root, dirs, files in walk(r"CroppedYale"):
    count = 0
    train_data_are_set = False  # Check if training data are all prepared
    for f in files:
        if count >= 35:
            train_data_are_set = True  # Training data are prepared when the number of image is over 35
        if f[-3:] != "pgm":  # Ignore files are not in .pgm format
            continue
        img = io.imread(join(root, f))  # Read image
        img = color.rgb2gray(img)  # Convert image to gray
        if not train_data_are_set:  # Add file name, nd-array form and label
            train_data.append((f, np.array(img, dtype=float), int(f[5:7])))
        else:  # Add file name, nd-array form and spaces for label
            test_data.append((f, np.array(img, dtype=float), (None, None)))
        count += 1

SAD_result = 0
SSD_result = 0

for i, test_image in enumerate(test_data):  # Calculate SAD, SSD
    SAD, SSD = [], []
    for train_image in train_data:
        z = abs(train_image[1] - test_image[1])
        SAD.append(np.sum(z))
        SSD.append(np.sum(z ** 2))
    # Find minimum SAD, SSD
    test_image = (test_image[0], (train_data[SAD.index(min(SAD))][2], train_data[SSD.index(min(SSD))][2]))
    if test_image[1][0] == int(test_image[0][5:7]):
        SAD_result += 1
    print("SAD Accuracy: ", round((SAD_result / (i+1)) * 100), "%")  # Accuracy of SAD classification
    if test_image[1][1] == int(test_image[0][5:7]):
        SSD_result += 1
    print("SSD Accuracy: ", round((SSD_result / (i+1)) * 100), "%")  # Accuracy of SSD classification

print("Final SAD accuracy: ", round((SAD_result / len(test_data)) * 100), "%")
print("Final SSD accuracy: ", round((SSD_result / len(test_data)) * 100), "%")
