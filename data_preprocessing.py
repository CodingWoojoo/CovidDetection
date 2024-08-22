from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm

scan_types = ['COVID','non-COVID']
covid_scans = 'data_set'
train_set = os.path.join(covid_scans)

def data_cleaning(scan_types):
    training_scans = []
    for defects_id, sp in enumerate(scan_types):
        for file in os.listdir(os.path.join(train_set, sp)):
            training_scans.append(['{}/{}'.format(sp, file), defects_id, sp])
    train_var = pd.DataFrame(training_scans, columns=['file', 'scanid','scan_types'])
    return train_var
    
data_cleaned = data_cleaning(scan_types)

def data_randomization(data_cleaned):
    data_cleaned = data_cleaned.sample(frac=1, random_state=42) 
    data_cleaned.index = np.arange(len(data_cleaned)) 
    return data_cleaned

data_random = data_randomization(data_cleaned)

IMAGE_SIZE = 224
def input_image(filepath):
    return cv2.imread(os.path.join(train_set, filepath)) 
    
def resize_scan_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
X_train_data = np.zeros((data_random.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(data_random['file'].values)):
    image = input_image(file)
    if image is not None:
        X_train_data[i] = resize_scan_image(image, (IMAGE_SIZE, IMAGE_SIZE))
X_Train_data = X_train_data / 255.
Y_train = data_random['scanid'].values
Y_train = to_categorical(Y_train, num_classes=2)
BATCH_SIZE = 32
EPOCHS = 7
SIZE=224
N_ch=3
X_train_data, X_val, Y_train, Y_val = train_test_split(X_Train_data, Y_train, test_size=0.2, random_state=42)
