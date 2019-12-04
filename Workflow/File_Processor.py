import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def write_table(dirname, filename):
    path = dirname + filename
    train_accuracies, val_accuracies, test_accuracies, train_loss, val_loss, test_loss = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5), unpack=True)

    best_test = 0
    best_idx = 0
    for i, acc in enumerate(test_accuracies):
        if acc > best_test:
            best_test = acc
            best_idx = i

    num_unfroze_idx = filename.find('_num_unfroze')
    wd_idx = filename.find("_wd")
    color_idx = filename.find("_COLORED")
    csv_idx = filename.find(".csv")

    lr = filename[3:num_unfroze_idx]
    num_unfroze = filename[num_unfroze_idx + len('_num_unfroze') + 1]

    if color_idx != -1:
        return
        # color = "Yes"
        wd = filename[wd_idx + 4: color_idx]
    else:
        # color = "No"
        wd = filename[wd_idx + 4: csv_idx]

    fields=[lr,num_unfroze,wd, str(train_accuracies[best_idx]), str(val_accuracies[best_idx]), str(test_accuracies[best_idx]), str(train_loss[best_idx]), str(val_loss[best_idx]), str(test_loss[best_idx])]
    
    with open(table_filename,'a+') as fd:
        writer = csv.writer(fd)
        writer.writerow(fields)
    
# experiments_dir = "/Users/shawn/Documents/GitHub/visual-guitar-chord-classifier/Workflow/experiments/csv_files/ResNet/"
# table_filename = experiments_dir + "ult_table.csv"

# for filename in os.listdir(experiments_dir):
#     print (filename)
#     if filename != "conf_mat":
#         write_table(experiments_dir, filename)

def plot_accuracies(dirname, filename):
    path = dirname + filename
    train_accuracies, val_accuracies, test_accuracies = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0,1,2), unpack=True)
    epochs = np.arange(1, len(train_accuracies) + 1)
    training, = plt.plot(epochs, train_accuracies * 100., color="red", label="Train") 
    validation, = plt.plot(epochs, val_accuracies * 100., color="blue", label="Validation")
    testing, = plt.plot(epochs, test_accuracies * 100., color="green", label="Test")
    plt.legend(loc="lower right")

    axes = plt.axes()
    axes.set_ylim([0, 105])
    
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epochs")

    plt.savefig(dirname + "plots/" + filename[:-4] + "_acc_plot.png")
    plt.clf()

experiments_dir = "/Users/shawn/Documents/GitHub/visual-guitar-chord-classifier/Workflow/experiments/csv_files/ResNet/"
filename = "lr=0.001_num_unfroze=2_epochs=100_wd=0_COLORED.csv"
plot_accuracies(experiments_dir, filename)