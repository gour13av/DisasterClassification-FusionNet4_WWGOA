import numpy as np
from unet import dilated_unet
import cv2
from sklearn.metrics import mutual_info_score
from save_load import load
from classification import FusionNet_4



def fitness(x):
    soln = np.round(x)
    soln = abs(soln)
    img = cv2.imread('./Dataset/RescueNet/Disaster/10866.jpg')
    img = cv2.resize(img,(200, 200), cv2.INTER_AREA)
    inpu_shape = img.shape
    unet = dilated_unet(inpu_shape, soln)
    img = np.expand_dims(img, axis=0)
    predicted_mask = unet.predict(img)

    img = img.reshape(img.shape[1], img.shape[2], img.shape[3])

    true_mask = img
    true_mask_flat = true_mask.flatten()
    predicted_mask_flat = predicted_mask.flatten()

    # Calculate Mutual Information
    mi = mutual_info_score(true_mask_flat, predicted_mask_flat)

    return 1 / mi


def objective_func_70(x):
    x_train = load('x_train_70.0')
    y_train = load('y_train_70.0')
    x_test = load('x_test_70.0')
    y_test = load('y_test_70.0')

    soln = np.round(x)
    selected_indices = np.where(soln == 1)[0]

    x_train = x_train[:, selected_indices]
    x_test = x_test[:, selected_indices]

    pred, met = FusionNet_4(x_train, y_train, x_test, y_test)
    fit = met[0]
    return fit


def objective_func_80(x):
    x_train = load('x_train_80.0')
    y_train = load('y_train_80.0')
    x_test = load('x_test_80.0')
    y_test = load('y_test_80.0')

    soln = np.round(x)
    selected_indices = np.where(soln == 1)[0]

    x_train = x_train[:, selected_indices]
    x_test = x_test[:, selected_indices]

    pred, met = FusionNet_4(x_train, y_train, x_test, y_test)

    fit = met[0]
    return fit
