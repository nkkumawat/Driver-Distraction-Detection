import app.Utils as utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from pylab import savefig

def confusion_matrix_fn( model):
    train_data_npy = np.load(utils.TRAIN_DATA_COLOR_NPY, encoding="latin1")
    test_data = train_data_npy[-1000:]
    total = 0
    real = 0
    confusion_matrix =  [ [0] * 10 for _ in range(10)]
    for num, data in enumerate(test_data[:]):
        # print(num , data)
        img_num = np.argmax(data[1])
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(utils.IMG_SIZE, utils.IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        confusion_matrix[img_num][np.argmax(model_out)] += 1
        if(np.argmax(model_out) == img_num):
            real +=1
        total +=1

    print("correct pics : " , real, "total pics : ",total,"accuracy : ",float(real*100)/total)
    for i in range(10):
        print(confusion_matrix[i])

    class_wise_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0]
    for i in range(10):
        sum = 0
        for j in range(10):
            sum += confusion_matrix[i][j] 
        class_wise_acc[i] =   ( float(confusion_matrix[i][i] )/ sum ) * 100

    fig, ax = plt.subplots(figsize=(4, 4))
    plt.ylabel('Accuracy ( % )')
    plt.xlabel('Classes')
    plt.bar(np.arange(10), class_wise_acc)
    plt.xticks(np.arange(10), ('c0', 'c1', 'c2', 'c3','c4', 'c5', 'c6', 'c7', 'c8', 'c9'))
    fig.savefig('graphs/clsacc.png')

    plt.gcf().clear()
    df_cm = pd.DataFrame(confusion_matrix)
    # svm = sn.heatmap(df_cm)
    svm = sn.heatmap(df_cm, annot=False,cmap='coolwarm', linecolor='white', linewidths=1)
    plt.savefig('graphs/conmtrx.png')
    print("Confusion Matrix Done ...........................................")
    return confusion_matrix , class_wise_acc
