import csv
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
import warnings
import random
import math

#convert txt file to csv
with open('heartdisease.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('heartdisease.csv', 'w',newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca','thal', 'num'))
        writer.writerows(lines)
 
warnings.filterwarnings("ignore")
       
# Example of Naive Bayes implemented from Scratch in Python

#calculating mean of column values belonging to one class
def mean(columnvalues):
    s=0
    n=float(len(columnvalues))
    for i in range(len(columnvalues)):
        s=s+float(columnvalues[i])
    return s/n

#calculating standard deviation of column values belonging to one class
def stdev(columnvalues):
    avg = mean(columnvalues)
    s=0.0
    num=len(columnvalues)
    for i in range(num):
        s=s+pow(float(columnvalues[i])-avg,2)
    variance = s/(float(num-1))
    return math.sqrt(variance)


# Reading CSV file
filename = 'heartdisease.csv'
lines = csv.reader(open(filename, "r"))
dataset = list(lines)
for i in range(len(dataset)-1):
    dataset[i] = [float(x) for x in dataset[i+1]]


for z in range(5):
    print("\n\n\nTest Train Split no. ",z+1,"\n\n\n")
    trainsize = int(len(dataset) * 0.75)
    trainset = []
    testset = list(dataset)
    for i in range(trainsize):
        index = random.randrange(len(testset))
        trainset.append(testset.pop(index))

    # separate list according to class
    classlist = {}
    for i in range(len(dataset)):
        class_num = float(dataset[i][-1])
        row=dataset[i]
        if (class_num not in classlist):
            classlist[class_num] = []
        classlist[class_num].append(row)
    
    # preparing data class wise
    class_data = {}
    for class_num, row in classlist.items():
        class_datarow = [(mean(columnvalues), stdev(columnvalues)) for columnvalues in zip(*row)]
        class_datrow=class_datarow[0:13]
        class_data[class_num] =class_datarow 
     
    # Getting test vector
    y_test=[]
    for j in range(len(testset)):
        y_test.append(testset[j][-1])    
        
    # Getting prediction vector
    y_pred = []
    for i in range(len(testset)):
        class_probability = {}
        for class_num, row in class_data.items():
            class_probability[class_num] = 1
            for j in range(len(row)):
                calculated_mean, calculated_dev = row[j]
                x = float(testset[i][j])
                if(calculated_dev!=0):
                    power =math.exp(-(math.pow(x-calculated_mean,2)/(2*math.pow(calculated_dev,2))))
                    probability= (1 / (math.sqrt(2*math.pi) *calculated_dev)) * power
                class_probability[class_num] *= probability

        resultant_class, max_prob = -1, -1
        for class_num, probability in class_probability.items():
            if resultant_class == -1 or probability > max_prob:
                max_prob = probability
                resultant_class = class_num 
        
        y_pred.append(resultant_class)
    
    # Getting Accuracy
    count = 0
    for i in range(len(testset)):
        if testset[i][-1] == y_pred[i]:
            count += 1
    accuracy=(count/float(len(testset))) * 100.0
    print("\n\n Accuracy: ",accuracy,"%")

    y1=[float(k) for k in y_test]
    y_pred1=[float(k) for k in y_pred]
    
    print("\n\n\n\nConfusion Matrix")
    cf_matrix=confusion_matrix(y1,y_pred1)
    print(cf_matrix)
    
    print("\n\n\n\nF1 Score")
    f_score = f1_score(y1,y_pred1,average='weighted')
    print(f_score)
    
    # Matrix from 1D array
    y2=np.zeros(shape=(len(y1),5))
    y3=np.zeros(shape=(len(y_pred1),5))
    for i in range(len(y1)):
        y2[i][int(y1[i])]=1
    
    for i in range(len(y_pred1)):
        y3[i][int(y_pred1[i])]=1
     
    
    # ROC Curve generation
    n_classes = 5
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y2[:, i], y3[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), y3.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    print("\n\n\n\nROC Curve")
    # First aggregate all false positive rates
    lw=2
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower right")
    plt.show()












