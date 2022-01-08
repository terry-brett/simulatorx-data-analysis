import cv2
from keras.models import load_model
import numpy as np
import glob
import matplotlib.pyplot as plt

def run_VGG16_model():
    list_of_images = glob.glob("../data/facial_images/*")

    # run VGG16 facial recognition model
    eth_model = load_model('../data/facial_features/eth-model.h5')
    gen_model = load_model('../data/facial_features/gen-model.h5')
    age_model = load_model('../data/facial_features/age-model.h5')

    img_arr = []
    path_list = []
    for image in list_of_images:
        img = cv2.imread(image)
        img = cv2.resize(img, (96, 96))
        img_arr.append(img)
        path_list.append(image)

    img_arr = np.asarray(img_arr)
    img_arr = img_arr / 255.0

    age_predicts = age_model.predict(img_arr)
    age_predicts = np.argmax(age_predicts, axis=1)

    gen_predicts = gen_model.predict(img_arr)
    gen_predicts = np.argmax(gen_predicts, axis=1)

    eth_predicts = eth_model.predict(img_arr)
    eth_predicts = np.argmax(eth_predicts, axis=1)

    gender_classes = ['male', 'female']
    age_classes = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                   '55-59', '60-64', '65-69', '70-74', '75+']
    ethinicity_classes = ['white', 'black', 'asian', 'other']

    # gender prediction figure
    plt.clf()
    plt.figure(figsize=(16, 16))
    for i in range(len(img_arr)):
        plt.subplot(3, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_arr[i], cmap=plt.cm.binary)
        plt.xlabel(gender_classes[gen_predicts[i]])
    plt.savefig('../results/gender_prediction.png')

    # ethinicity prediction figure
    plt.clf()
    plt.figure(figsize=(16, 16))
    for i in range(len(img_arr)):
        plt.subplot(3, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_arr[i], cmap=plt.cm.binary)
        plt.xlabel(ethinicity_classes[eth_predicts[i]])
    plt.savefig('../results/ethnicity_prediction.png')

    # age prediction figure
    plt.clf()
    plt.figure(figsize=(16, 16))
    for i in range(len(img_arr)):
        plt.subplot(3, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_arr[i], cmap=plt.cm.binary)
        plt.xlabel(age_classes[age_predicts[i]])
    plt.savefig('../results/age_prediction.png')