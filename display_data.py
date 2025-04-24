
import os
import matplotlib.pyplot as plt
from numpy import max, polyfit, poly1d
import numpy as np
from collections import defaultdict
import math

#Load a csv as a dict
def load_csv(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        keys = first_line.split(',')
        data = {key: [] for key in keys}

        for line in file:
            values = line.strip().split(',')
            values = [int(values[0])] + [float(x) for x in values[1:]]
            for key, value in zip(keys, values):
                data[key].append(value)
                
    return data

    
#built in max/avg didnt work due to weird float shenanigans but this did so whatever
def floatmax(*args):
    max_val = args[0]
    for val in args[1:]:
        if val > max_val:
            max_val = val
    return max_val
def floatavg(*args):
    total = 0.0
    count = 0
    for val in args:
        total += val
        count += 1
    return total / count if count > 0 else 0.0

#prints train/test acct and max attack accuracy for each model in model_list
def print_results(model_list):   
    for model in model_list:
        history_dict = load_csv(model + history_file_extension)
        accuracies_dict = load_csv(model + accuracies_file_extension)
        print(f"Model:{model}\nTrain/Test Acc:{history_dict['accuracy'][-1]} vs {history_dict['val_accuracy'][-1]}")
        aa2 = ([floatavg(accuracies_dict["xgb2"][i], accuracies_dict["knn2"][i], accuracies_dict["cnn2"][i],accuracies_dict["xgb1"][i], accuracies_dict["knn1"][i], accuracies_dict["cnn1"][i]) for i in range(len(accuracies_dict["xgb2"]))])
        
        print(f"Attack Accuracy:{aa2[-1]}\n")


def get_overfit_vs_attack_accuracy(model_list):
    model_overfit_list = []
    model_attack_accuracy_list = []
    for model in model_list:
        accuracies_dict = load_csv(model + accuracies_file_extension)
        history_dict = load_csv(model + history_file_extension)
        model_overfit_list.append([train - val for train, val in zip(history_dict["accuracy"], history_dict["val_accuracy"])])
        max_list = [floatmax(accuracies_dict["xgb1"][i],
                    accuracies_dict["knn1"][i],
                    accuracies_dict["cnn1"][i],
                    accuracies_dict["xgb2"][i],
                    accuracies_dict["knn2"][i],
                    accuracies_dict["cnn2"][i]) 
                for i in range(len(accuracies_dict["xgb1"]))]
        model_attack_accuracy_list.append(max_list)
    return model_overfit_list,model_attack_accuracy_list
    
def graph_overfit_vs_acc(model_list):
    overfit_gap,accuracys = get_overfit_vs_attack_accuracy(model_list)
    cols = math.ceil(math.sqrt(len(model_list)))
    rows = math.ceil(len(model_list) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    x_vals = list(range(100))
    axs = axs.flatten() 
    for i, (gap, acc) in enumerate(zip(overfit_gap, accuracys)):
        ax1 = axs[i]
        ax2 = ax1.twinx()
        ax2.plot(x_vals, gap, 'b.', label='Overfit Gap')
        ax1.plot(x_vals, acc, 'r.', label='Attack Accuracy')
        ax1.set_ylim(-.1, 1)
        ax2.set_ylim(-.1, 1)
        ax2.set_xlim(0, 99)
        ax2.set_title(model_list[i])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Overfit Gap", color='b')
        ax1.set_ylabel("Attack Accuracy", color='r')
        ax2.grid(True)
    for j in range(len(model_list), len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    plt.show()
    
def graph_history(model_list):
    cols = math.ceil(math.sqrt(len(model_list)))
    rows = math.ceil(len(model_list) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    x_vals = list(range(100))  # x-axis: 0 to 99
    axs = axs.flatten()

    for i, model in enumerate(model_list):
        history_dict = load_csv(model + history_file_extension)
        train_acc = history_dict["accuracy"]
        test_acc = history_dict["val_accuracy"]
        ax1 = axs[i]
        ax2 = ax1.twinx()
        ax1.plot(x_vals, train_acc, 'g.', label='Train Accuracy')
        ax2.plot(x_vals, test_acc, 'm.', label='Test Accuracy')
        ax1.set_ylim(-.1, 1)
        ax2.set_ylim(-.1, 1)
        ax2.set_xlim(0, 99)
        ax2.set_title(model)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Accuracy", color='g')
        ax2.set_ylabel("Test Accuracy", color='m')
        ax2.grid(True)

    for j in range(len(model_list), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

#this is basic code used to generate figure 2, with some model dicts being pulled directly
def figure_2(all_models):
    max_list_1 = []
    max_list_2 = []
    average_list_1 = []
    average_list_2 = []
    for model in all_models:
        history_dict = load_csv(model + history_file_extension)
        accuracies_dict = load_csv(model + accuracies_file_extension)
        max_list_1.append([floatmax(accuracies_dict["xgb1"][i],accuracies_dict["knn1"][i],accuracies_dict["cnn1"][i])for i in range(len(accuracies_dict["xgb1"]))])
        max_list_2.append([floatmax(accuracies_dict["xgb2"][i],accuracies_dict["knn2"][i],accuracies_dict["cnn2"][i]) for i in range(len(accuracies_dict["xgb1"]))])
        average_list_1.append([floatavg(accuracies_dict["xgb1"][i],accuracies_dict["knn1"][i],accuracies_dict["cnn1"][i])for i in range(len(accuracies_dict["xgb1"]))])
        average_list_2.append([floatavg(accuracies_dict["xgb2"][i],accuracies_dict["knn2"][i],accuracies_dict["cnn2"][i]) for i in range(len(accuracies_dict["xgb1"]))])
        
    max1 = np.mean(np.array(max_list_1), axis=0)
    max2 = np.mean(np.array(max_list_2), axis=0)
    average1 = np.mean(np.array(average_list_1), axis=0)
    average2 = np.mean(np.array(average_list_2), axis=0)
    acc_dict = load_csv('CNN_90percent_data_without_Datagen-100epoch_attack_accuracies.csv')
    epochs = list(range(100))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Max attacks plot
    axs[2].plot(epochs, max1, label='Max Attack 1')
    axs[2].plot(epochs, max2, label='Max Attack 2')
    axs[2].set_title("Max Attack Accuracy")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].legend()
    axs[2].grid()

    # Average attacks plot
    axs[1].plot(epochs, average1, label='Average Attack 1')
    axs[1].plot(epochs, average2, label='Average Attack 2')
    axs[1].set_title("Average Attack Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid()

    # Individual attacks plot
    attack_epochs = acc_dict['epoch']
    axs[0].plot(attack_epochs, acc_dict['xgb1'], label='XGB1', color='blue')
    axs[0].plot(attack_epochs, acc_dict['knn1'], label='KNN1', color='skyblue')
    axs[0].plot(attack_epochs, acc_dict['cnn1'], label='MLP', color='navy')
    axs[0].plot(attack_epochs, acc_dict['xgb2'], label='XGB2', color='red')
    axs[0].plot(attack_epochs, acc_dict['knn2'], label='KNN2', color='pink')
    axs[0].plot(attack_epochs, acc_dict['cnn2'], label='MLP2', color='darkred')
    axs[0].set_title("All Attacks")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid()

    #I was also interested in differences between 1 and 2, this part gets that
    max1 = np.array(max1)
    max2 = np.array(max2)
    avg1 = np.mean(max1)
    avg2 = np.mean(max2)
    avg_diff = avg1 - avg2
    percent_diff = (avg_diff / avg2) * 100
    print(f"Avg Max Attack 1{avg1:.4f}")
    print(f"Avg Max Attack 2{avg2:.4f}")
    print(f"Difference: {avg_diff:.4f}")
    print(f"Relative Difference: {percent_diff:.2f}%")

    plt.tight_layout()
    plt.show()
    
    
    
#All models:
all_models = [file[:-12] for file in os.listdir() if file.endswith(".keras")]

#dicts are named with model name plus string
accuracies_file_extension = "_attack_accuracies.csv"
history_file_extension = "_history.csv"

#All 90% models
models_90percent = [model for model in all_models if "90" in model]
#All 75% models
models_75percent = [model for model in all_models if "75" in model]
#All 50% models
models_50percent = [model for model in all_models if "50" in model]
#All Resnet models
models_resnet = [model for model in all_models if "Resnet" in model]
#All CNN models
models_cnn = [model for model in all_models if "CNN" in model]
#All with datagen
models_with = [model for model in all_models if "with" in model and "without" not in model]
#All without datagen
models_without = [model for model in all_models if "without" in model]



#check - 3x1 training % of overfit on x and attack acc on y
#check - 2x1 with/without data gen of overfit on x and attack acc on y
#check - 2x1 Cnn/Resnet of overfit on x and attack acc on y

#Average overfit and average accuracy on y and epoch on x

#All Model attack1 vs attack 2
    #all attack 1 vs all attack 2
    #Average attack 1 vs average attack 2
    #Max attack1 vs max attack 2
    #Best fit of Max attack 1 vs max attack 2



print_results(all_models)
graph_overfit_vs_acc(all_models)
graph_history(all_models)



