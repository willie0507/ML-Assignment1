from skimage import io, transform, color
# import numpy as np
from os import walk
from os.path import join
import tensorflow as tf

train_data = []  # (filename, label)
test_data = []
test_data_SAD = []
test_data_SSD = []
train_data_array = []  # All training data in nd-array form
test_data_array = []

# Split into training data and testing data
for root, dirs, files in walk(r"../CroppedYale"):
    count = 0
    train_data_are_set = False  # Check if training data are all prepared
    for f in files:
        if count >= 35:
            train_data_are_set = True  # Training data are prepared when the number of image is over 35
        if f[-3:] != "pgm":  # Ignore files are not in .pgm format
            continue
        img = io.imread(join(root, f))  # Read image
        img = transform.resize(img, (224, 224), mode='constant')  # Resize image
        img = color.rgb2gray(img)  # Convert image to gray
        if not train_data_are_set:
            train_data.append((f, int(f[5:7])))  # Add file name and label
            train_data_array.append(tf.constant(img))  # Add image in nd-array form
        else:
            test_data.append(f)  # Add file name
            test_data_array.append(tf.constant(img))  # Add image in nd-array form
        count += 1

with tf.Session() as sess:
    test_data_array = sess.run(test_data_array)
    train_data_array = sess.run(train_data_array)

# Calculate SAD and SSD for each image
with tf.Session() as sess:
    for i, test_image in enumerate(test_data_array):  # Calculate SAD, SSD
        SAD = []
        SSD = []
        for train_image in train_data_array:
            z = tf.abs((tf.div(train_image, 100) - tf.div(test_image, 100)))
            z = tf.multiply(z, 100)
            SAD_result = tf.reduce_sum(z)
            SSD_result = tf.reduce_sum(tf.square(z))
            SAD_result = sess.run(SAD_result)
            SSD_result = sess.run(SSD_result)
            SAD.append(SAD_result)
            SSD.append(SSD_result)

        print('SAD: ', SAD)
        print('SSD: ', SSD)

'''    
    test_data_SAD.append((test_data[i], train_data[SAD.index(min(SAD))][1]))  # Find the minimum SAD
    test_data_SSD.append((test_data[i], train_data[SSD.index(min(SSD))][1]))  # Find the minimum SSD
    # test_data[i] = (test_data[i], train_data[SAD.index(min(SAD))][1])

for t_data in test_data:
    print(t_data)
'''