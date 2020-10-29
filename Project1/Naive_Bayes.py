import numpy as np
from PIL import Image
import sys

def Class_separation(data):
    separated = {}
    for i in range(len(data)):
        vector = data[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
 
def PCA(a, comp):
    a = a - np.mean(a, axis=0)
    co_mat = np.cov(a, rowvar = False)
    E, V = np.linalg.eigh(co_mat)
    E_idx = np.argsort(E)[::-1]
    values = E[E_idx]
    vectors = V[:,E_idx]
    pca = np.dot(a,vectors)[:,:comp]
    return pca

def summarize(data):
    mean_stdev = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*data)]
    del mean_stdev[-1]
    return mean_stdev
 
def class_summary(data):
    separated = Class_separation(data)
    mean_stdev = {}
    for class_val, instances in separated.items():
        mean_stdev[class_val] = summarize(instances)
    return mean_stdev

def loadData_test(filename):
    images = []
    with open(filename) as f:
        lines = f.readlines()
    lines = np.asarray(lines)
    for l in lines:
        x = l.strip()
        img = Image.open(x).convert('L')
        img = np.asarray(img.resize((w,h), Image.NEAREST)).flatten()
        images.append(img)
    images = np.asarray(images)
    pca = PCA(images,32)
    return pca
 
def cal_Prob(x, mean, stdev):
    expo_val = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
    return (1 / (np.sqrt(2*np.pi) * stdev)) * expo_val
 
def cal_Class_Prob(mean_stdev, input_vector):
    prob = {}
    for class_val, class_summaries in mean_stdev.items():
        prob[class_val] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            prob[class_val] *= cal_Prob(x, mean, stdev)
    return prob
            
def predict(mean_stdev, inputVector):
    prob = cal_Class_Prob(mean_stdev, inputVector)
    best_label, best_prob = None, -1
    for class_val, probability in prob.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_val
    return best_label
 
def Predict_test(mean_std, test_Set):
    predictions = []
    for t in range(len(test_Set)):
        res = predict(mean_std, test_Set[t])
        predictions.append(res)
    return predictions
 
def Accuracy_train(test_Set, pred):
    right = 0
    for t in range(len(test_Set)):
        if test_Set[t][-1] == pred[t]:
            right += 1
    return (right/float(len(test_Set))) * 100.0

def loadData_train(filename):
    label = []
    images = []
    with open(filename) as f:
        lines = f.readlines()
    lines = np.asarray(lines)
    for l in lines:
        x, y = l.split(' ')
        label.append(y.strip())
        img = Image.open(x).convert('L')
        img = np.asarray(img.resize((w,h), Image.NEAREST)).flatten()
        images.append(img)
    images = np.asarray(images)
    label = np.asarray(label)
    pca = PCA(images,32)
    return pca, label

filename_train= sys.argv[1]
filename_test = sys.argv[2]

#filename_train = r'C:\Python36\mini-project-1\train.txt'
#filename_test = r'C:\Python36\mini-project-1\test.txt'

w = 32
h = 32
x, y = loadData_train(filename_train)
X = loadData_test(filename_test)
y2 = set(y)
y3 = {}
summ = 0
for a in y2:
    y3[a] = summ
    summ += 1
y4 = {v: k for k, v in y3.items()}
y5 = np.zeros(len(y))
for i in range(len(y)):
    y5[i] = y3[y[i]]
y = y5.reshape(-1,1)
dataset = np.hstack((x,y))
train_Set = dataset
test_Set = X
mean_var = class_summary(train_Set)
pred = Predict_test(mean_var, train_Set)
for a in pred:
    print(y4[a])
accuracy = Accuracy_train(train_Set, pred)
print(accuracy)
