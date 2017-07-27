import os
import glob
import random
import pickle

import cv2
import numpy as np

dir_data = "I:\\test"
dir_types = ["Type_1", "Type_2", "Type_3"]

file_names = glob.glob(os.path.join(dir_data, "*.jpg"), recursive=True)
random.shuffle(file_names)

print("read the images")

if not os.path.exists("tests.pkl"):
    images = []
    for file_name in file_names:

        image = cv2.imread(file_name)
        image = cv2.resize(image, (299, 299))
        image.astype(np.float32)

        images.append(image)
        print("{} of {}".format(len(images), len(file_names)))
    with open('tests.pkl', mode='wb') as f:
        pickle.dump(images, f)

if os.path.exists("tests.pkl"):
    with open("tests.pkl", mode="rb") as f:
        images = pickle.load(f)

X_test = np.array(images)

from keras.models import *

model = load_model("model.h5")
preds = model.predict(X_test)

with open("result.csv", "w") as f:
    f.write("image_name,Type_1,Type_2,Type_3\n")
    for file_name, pred in zip(file_names, preds):
        f.write(os.path.split(file_name)[-1] + ",")
        count = 0
        for item in pred:
            if count == len(pred) - 1:
                f.write(str(item))
            else:
                f.write(str(item) + ",")
            count += 1
        f.write("\n")
