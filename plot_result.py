import matplotlib.pyplot as plt
import numpy as np
from save_load import load, save
import pandas as pd

def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learning Rate (%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learning Rate (%)',
            kind='bar',
            stacked=False)


    plt.ylabel(metric)
    plt.legend(loc='upper right')
    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)

def plotres():

    # 80, 20 variation
    vgg19_80 = load('vgg19_80')
    efficientnet_80 = load('efficientnet_80')
    resnet50_80 = load('resnet50_80')
    densenet121_80 = load('densenet121_80')
    proposed_80 = load('proposed_80')

    data = {
        'Vgg19': vgg19_80,
        'EfficientNet': efficientnet_80,
        'ResNet50': resnet50_80,
        'DenseNet121': densenet121_80,
        'PROPOSED': proposed_80
    }

    ind = ['MSE', 'MAE', 'NMSE', 'RMSE', 'MAPE']
    table = pd.DataFrame(data, index=ind)
    save('table1', table)
    tab = table.to_excel('./Results/table_80.xlsx')

    val1 = np.array(table)

    # learn rate 70, 30
    vgg19_70 = load('vgg19_70')
    efficientnet_70 = load('efficientnet_70')
    resnet50_70 = load('resnet50_70')
    densenet121_70 = load('densenet121_70')
    proposed_70 = load('proposed_70')


    data1 = {
        'Vgg19': vgg19_70,
        'EfficientNet': efficientnet_70,
        'ResNet50': resnet50_70,
        'DenseNet121': densenet121_70,
        'PROPOSED': proposed_70
    }

    ind = ['MSE', 'MAE', 'NMSE', 'RMSE', 'MAPE']
    table1 = pd.DataFrame(data1, index=ind)
    save('table2', table1)
    tab = table1.to_excel('./Results/table_70.xlsx')

    val2 = np.array(table1)

    method = ["Vgg19", "EfficientNet", "ResNet50", "DenseNet121", "PROPOSED"]
    metrices_plot = ['MSE', 'MAE', 'NMSE', 'RMSE', 'MAPE']
    metrices = [val2, val1]
    save('met', metrices)

    for i in range(len(metrices_plot)):
        bar_plot(method, metrices[0][i, :], metrices[1][i, :],
                 metrices_plot[i])

    for i in range(2):
        print('Metrices-Dataset--' + str(i + 1))
        tab = pd.DataFrame(metrices[i], index=metrices_plot, columns=method)
        print(tab)


