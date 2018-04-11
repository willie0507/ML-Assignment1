Method description
-
Load training data in format (filename, nd-array form, label), load testing data in format
(filename, nd-array form, (SAD label, SSD label)).

    train_data = []  # (filename, nd-array form, label)
    test_data = []  # (filename, nd-array form, (SAD label, SSD label))

For each group, split 35 images to training image, the rest of the images to testing image, and covert to nd-array.

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

Calculate the SAD (sum of absolute distance) and SSD (sum of square distance) for each test image and train image.

    z = abs(train_image[1] - test_image[1])
    SAD.append(np.sum(z))
    SSD.append(np.sum(z ** 2))

Classify each testing image depends on the minimum SAD and SSD.

    # Find minimum SAD, SSD
    test_image = (test_image[0], (train_data[SAD.index(min(SAD))][2], train_data[SSD.index(min(SSD))][2]))

Calculate SAD and SSD Accuracy.

    print("Final SAD accuracy: ", round((SAD_result / len(test_data)) * 100), "%")
    print("Final SSD accuracy: ", round((SSD_result / len(test_data)) * 100), "%")
    
Experimental results - accuracy
-
    Final SAD accuracy:  45 %
    Final SSD accuracy:  46 %

Discussion of difficulty or problem encountered
-
At the beginning, I just simply use the code which TA had provided, but it cause long time (about 20 minutes)
for the execution time. I found the code which provided by TA try to convert the nd-array in floating point by using
array divide 100 and than multiply 100. It will cause a lot of time to calculate.

    im1 = io.imread('1.pgm')
    im2 = io.imread('2.pgm')
    x = np.array(im1)
    y = np.array(im2)
    z = abs(x/100 - y/100)
    z = z*100
    SAD = np.sum(z)
    SSD = np.sum(z**2)
    
To solve this problem, I set dtype to float when loading images to nd-array form.

    if not train_data_are_set:  # Add file name, nd-array form and label
        train_data.append((f, np.array(img, dtype=float), int(f[5:7])))
    else:  # Add file name, nd-array form and spaces for label
        test_data.append((f, np.array(img, dtype=float), (None, None)))
        
After doing this, the execution time reduce to 4 minutes.