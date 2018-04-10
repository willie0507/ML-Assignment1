from skimage import io, transform, color
import numpy as np
from os import walk
from os.path import join

train_data = []  # (feature, label)
test_data = []
test_data_SAD = []
test_data_SSD = []
train_data_array = []  # All training data in nd-array form
test_data_array = []

for root, dirs, files in walk(r"../CroppedYale"):
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
            train_data.append((f, int(f[5:7])))  # Add file name and label
            train_data_array.append(np.array(img))  # Add image in nd-array form
        else:
            test_data.append(f)  # Add file name
            test_data_array.append(np.array(img))  # Add image in nd-array form
        count += 1

for i, test_image in enumerate(test_data_array):  # Calculate SAD, SSD
    SAD = []
    SSD = []
    for train_image in train_data_array:
        z = abs(train_image/100 - test_image/100)
        z = z*100
        SAD.append(np.sum(z))
        SSD.append(np.sum(z**2))
    test_data_SAD.append((test_data[i], train_data[SAD.index(min(SAD))][1]))  # Find the minimum SAD
    test_data_SSD.append((test_data[i], train_data[SSD.index(min(SSD))][1]))  # Find the minimum SSD

result = 0
for t_data in test_data_SAD:  # Calculate SAD Acc
    if t_data[1] == int(t_data[0][5:7]):
        result += 1

print("SAD Acc: ", result/len(test_data))

result = 0
for t_data in test_data_SSD:  # Calculate SSD Acc
    if t_data[1] == int(t_data[0][5:7]):
        result += 1

print("SSD Acc: ", result/len(test_data))
