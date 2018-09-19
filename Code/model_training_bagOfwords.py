#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:34:44 2018

@author: amankumar
"""

# Importing the Modules and Libraries
import numpy as np
import matplotlib.pyplot as plt

# classifiers
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score





# Defining significant variables
dataset = "IMDB"

# vocabulary size
size_v = 10000

# binary or frequency
bow_model="binary"
    
# micro or macro or whatever
f1_avg_param = "micro"

# params range

depth_from, depth_to, depth_step =  1, 2e3, 2

alpha_from, alpha_to, alpha_step = 1e-6, 1e3, 10
c_from, c_to, c_step = 1e-6, 1e3, 10

# Defining Dictionaries
# Defining a dictionary that contains real paths 
path={
    "IMDB":{
        "Training":"IMDB-train-refined.txt",
        "Valid":"IMDB-valid-refined.txt",
        "Test":"IMDB-test-refined.txt",
    }
}
vector_binary_bow = {
    "Training":[],
    "Valid":[],
    "Test":[]
}


for dataSetType in path[dataset]:
    
    # Read file 
    training_file = open(path[dataset][dataSetType],"r") 
    content_list = training_file.read().split('\n')[:-1]
    content_list[:] = [row.split('\t') for row in content_list]   
    

    binary_bow = np.zeros((len(content_list), size_v+1))
    
    

    # making vector representations
    for i in range(len(content_list)):
        total = 0
        content_list[i][0] = content_list[i][0].split(' ')
        content_list[i][0] = [int(word_id) for word_id in content_list[i][0]]
        content_list[i][1] = int(content_list[i][1])
        
        binary_bow[i,size_v] = content_list[i][1]
    
        for eachId in content_list[i][0]:
            binary_bow[i,eachId-1] = 1   
        
               
    # saving the vector representation
    vector_binary_bow[dataSetType] = binary_bow
  
    
# Creating the vectors
x_train,y_train = vector_binary_bow["Training"][:,:-1],np.squeeze(vector_binary_bow["Training"][:,-1:])
x_valid,y_valid = vector_binary_bow["Valid"][:,:-1],np.squeeze(vector_binary_bow["Valid"][:,-1:])
x_test,y_test = vector_binary_bow["Test"][:,:-1],np.squeeze(vector_binary_bow["Test"][:,-1:])

def print_all_scores(classfier):
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_train)
    print("Training set F1 Score : \t" + str(f1_score(y_train, y_pred, average=f1_avg_param)))
    y_pred = classifier.predict(x_valid)
    print("Validation set F1 Score : \t" + str(f1_score(y_valid, y_pred, average=f1_avg_param)))
    y_pred = classifier.predict(x_test)
    print("Test set F1 Score : \t\t" + str(f1_score(y_test, y_pred, average=f1_avg_param)))    
    


def performance_plot(X,Y,optimal_point,classifier_name,param_name): 
    plt.plot(X, Y,'y')
    plt.plot(X, Y,'g.')
    plt.plot(optimal_point[0], optimal_point[1],'ro')
    plt.ylabel('F1 Score')
    plt.xlabel('Log ('+param_name+')')
    plt.title('Plot - Validation Set Performance of '+ classifier_name + ' w.r.t. '+ param_name)
    plt.show()
    
    
# Naive bayes classifier


print("Bernoulli Naive Bayes Classifier")
# Tuning Hyper Parameter alpha
hp_f1 = [] 
a=alpha_from
while a<alpha_to :
    classifier = BernoulliNB(alpha=a)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_valid)
    score = f1_score(y_valid, y_pred, average=f1_avg_param)
    hp_f1.append([math.log10(a),score,a])
    print("Alpha " + str(a) + " : " + str(score))
    a *= alpha_step
    # select alpha
selected_alpha = max(hp_f1, key=lambda item: item[1])
print("Alpha with best performance : "+ str(selected_alpha[2]))

#plot the graph
performance_plot(np.asarray(hp_f1)[:,0],np.asarray(hp_f1)[:,1],selected_alpha,"Naive Bayes classifier","Alpha")

#Training the classifier on the selected alpha
classifier = BernoulliNB(alpha=selected_alpha[2])
print_all_scores(classifier)



# Decision tree classifier
print("Decision Tree Classifier")

# Tuning Hyper Parameter alpha
hp_f1 = [] 
depth=depth_from
while depth<depth_to : 
    classifier = DecisionTreeClassifier(max_depth=depth)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_valid)
    score = f1_score(y_valid, y_pred, average=f1_avg_param)
    hp_f1.append([math.log10(depth),score,depth])
    print("Depth " + str(depth) + " : " + str(score))
    depth *= depth_step
    
    
# select depth
selected_depth = max(hp_f1, key=lambda item: item[1])
print("Max_depth with best performance : "+ str(selected_depth[2]))

#plot the graph
performance_plot(np.asarray(hp_f1)[:,0],np.asarray(hp_f1)[:,1],selected_depth,'Decision Tree Classifier','Max Depth')

classifier = DecisionTreeClassifier(max_depth=selected_depth[2])
print_all_scores(classifier)



# Support Vector Machine classification
print("Linear Support Vector Machine")

# Tuning Hyper Parameter alpha
hp_f1 = [] 
c=c_from
while c < c_to:    
    classifier = LinearSVC(C=c)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_valid)
    score = f1_score(y_valid, y_pred, average=f1_avg_param)
    hp_f1.append([math.log10(c),score,c])
    print("Penality " + str(c) + " : " + str(score))
    c*=c_step

# select penality
selected_c = max(hp_f1, key=lambda item: item[1])
print("Penality coeff with best performance : " + str(selected_c[2]))

#plot the graph
performance_plot(np.asarray(hp_f1)[:,0],np.asarray(hp_f1)[:,1],selected_c,'Linear SVM','Penality Coeff. C')

classifier = LinearSVC(C=selected_c[2])
print_all_scores(classifier)