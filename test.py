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
        if not train_data_are_set:
            train_data.append((f, np.array(img), int(f[5:7])))  # Add file name, nd-array form and label
        else:
            test_data.append((f, np.array(img), (None, None)))  # Add file name, nd-array form and spaces for label
        count += 1

for test_image in test_data:  # Calculate SAD, SSD
    SAD = []
    SSD = []
    for train_image in train_data:
        z = abs(train_image[1]/100 - test_image[1]/100)
        z = z*100
        SAD.append(np.sum(z))
        SSD.append(np.sum(z**2))
    # Find minimum SAD, SSD
    test_image = (test_image, (train_data[SAD.index(min(SAD))][2], train_data[SSD.index(min(SSD))][2]))

SAD_result = 0
SSD_result = 0
for t_data in test_data:  # Calculate SAD Acc
    if t_data[1][0] == int(t_data[0][5:7]):
        SAD_result += 1
    if test_data[1][1] == int(t_data[0][5:7]):
        SSD_result += 1

print("SAD Acc: ", SAD_result/len(test_data))
print("SSD Acc: ", SSD_result/len(test_data))
