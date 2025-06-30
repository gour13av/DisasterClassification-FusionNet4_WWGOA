from datagen import datagen
from save_load import load, save
import numpy as np
from WOWWP import Wowwp
from Objective_function import objective_func_70, objective_func_80
from classification import FusionNet_4, vgg19, EfficientNet, resnet50, densenet_121
from plot_result import plotres
import matplotlib.pyplot as plt


def full_analysis():
    datagen()
    # learning rate 70
    x_train_70 = load('x_train_70.0')
    x_test_70 = load('x_test_70.0')
    y_train_70 = load('y_train_70.0')
    y_test_70 = load('y_test_70.0')
    # learning rate 80
    x_train_80 = load('x_train_80.0')
    x_test_80 = load('x_test_80.0')
    y_train_80 = load('y_train_80.0')
    y_test_80 = load('y_test_80.0')

    learn_data = [(x_train_70, y_train_70, x_test_70, y_test_70, objective_func_70), (x_train_80, y_train_80, x_test_80, y_test_80, objective_func_80)]
    j = 70
    for i in learn_data:
        lb = np.zeros(i[0].shape[1])
        ub = np.ones(i[0].shape[1])
        pop_size = 6
        prob_size = len(lb)
        epochs = 100
        best_solution = Wowwp(i[-1], lb, ub, pop_size, prob_size, epochs)
        save('best_solution_'+str(j), best_solution)
        best_solution = load('best_solution_70')

        soln = np.round(best_solution)
        selected_indices = np.where(soln == 1)[0]

        x_train = i[0][:, selected_indices]
        y_train = i[1]
        x_test = i[2][:, selected_indices]
        y_test = i[3]

        pred, met = FusionNet_4(x_train, y_train, x_test, y_test)
        save('proposed_' + str(j), met)

        pred, met = vgg19(i[0], i[1], i[2], i[3])  # x_train, y_train, x_test, y_test
        save('vgg19_' + str(j), met)

        pred, met = EfficientNet(i[0], i[1], i[2], i[3])
        save('efficientnet_' + str(j), met)

        pred, met = resnet50(i[0], i[1], i[2], i[3])
        save('resnet50_' + str(j), met)

        pred, met = densenet_121(i[0], i[1], i[2], i[3])
        save('densenet121_' + str(j), met)
        
        j = 80

a = 0
if a == 1:
    full_analysis()

plotres()
plt.show()
