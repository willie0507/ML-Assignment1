from skimage import io, transform
import numpy as np
import os

'''
im1 = io.imread('1.pgm')
im2 = io.imread('2.pgm')

x = np.array(im1)
y = np.array(im2)
z = abs(x/100 - y/100)
z = z*100
SAD = np.sum(z)
SSD = np.sum(z**2)
print(SAD, SSD)
'''
train_data = []  # (feature, label)
test_data = []

for root, dirs, files in os.walk(r"..\CroppedYale"):
    count = 0
    train_data_allset = False
    for f in files:
        if count >= 35:
            train_data_allset = True
        if f[-3:] != "pgm":
            continue
        if not train_data_allset:
            train_data.append((f, f[5:7]))
        else:
            test_data.append(f)
        img = io.imread(os.path.join(root, f))
        img = transform.resize(img, (224, 224), mode='constant')  # resize all images
        count += 1
