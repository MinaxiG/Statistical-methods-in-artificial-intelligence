import numpy as np
from PIL import Image
import os
import sys

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
        img = np.asarray(img.resize((32,32), Image.NEAREST)).flatten()
        images.append(img)
    images = np.asarray(images)
    label = np.asarray(label)
    train_data = PCA(images,32)
    return train_data, label

def loadData_test(filename):
    images = []
    with open(filename) as f:
        lines = f.readlines()
    lines = np.asarray(lines)
    for l in lines:
        x = l.strip()
        img = Image.open(x).convert('L')
        img = np.asarray(img.resize((32,32), Image.NEAREST)).flatten()
        images.append(img)
    images = np.asarray(images)
    test_data = PCA(images,32)
    return test_data

def PCA(data, comp):
    data = data - np.mean(data, axis=0)
    co_mat = np.cov(data, rowvar = False)
    E, V = np.linalg.eigh(co_mat)
    E_idx = np.argsort(E)[::-1]
    values = E[E_idx]
    vectors = V[:,E_idx][:,:comp]
    pca = np.dot(data,vectors)
    mean_vect = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        mean_vect[i] =  np.mean(data.T[i])
    return pca

def getLoss(w,x,y):
    n = x.shape[0] 
    y_mat = matrix(y)
    scores = np.dot(x,w) 
    prob = soft_max(scores) 
    loss = (-1 / n) * np.sum(y_mat * np.log(prob)) 
    grad = (-1 / n) * np.dot(x.T,(y_mat - prob)) 
    return loss,grad

def matrix(Y):
    Y_matrix = np.zeros((len(Y),len(set(Y))))
    for i in range(len(Y)):
        for j in range(len(set(Y))):
            if(Y[i] == j):
                Y_matrix[i][j] = 1
    return Y_matrix

def soft_max(z):
    s_max = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return s_max

def predict(X):
    probs = soft_max(np.dot(X,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def cal_Accuracy(X,Y):
    probs,preds = predict(X)
    accuracy = sum(preds == Y)/(float(len(Y)))
    return accuracy*100.0

filename_train = sys.argv[1]
filename_test = sys.argv[2]

train_data,labels = loadData_train(filename_train)
test_data = loadData_test(filename_test)

encode_label = sorted(set(labels))
n_labels = {}
i = 0
for a in encode_label:
    n_labels[a] = i
    i = i + 1
y5 = np.zeros(len(labels))    
for i in range(len(labels)):
    y5[i] = n_labels[labels[i]]
o_labels = {v: k for k, v in n_labels.items()}
eta = 0.009
iterations = 1000
normalised_train_data = train_data / train_data.max(axis=0)
w = np.zeros([train_data.shape[1],len(np.unique(y5))])
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,normalised_train_data,y5)
    losses.append(loss)
    w = w - (eta * grad)
Accuracy = cal_Accuracy(train_data, y5)
print('Accuracy: ', Accuracy)
prob_test,pred_test = predict(test_data)
for i in range(len(pred_test)):
    print(o_labels[pred_test[i]])
