#!/usr/bin/env python
# coding: utf-8

'''
Authors    :     Chaithanya Pramodh Kasula, Srikaran Elakurthy, Aishwarya Varala
Description: AIT 726 - Homework 1. The following program is used to perform 
             logistic regression. It takes the 4 cleaned datasets, each as 
             specified in the homework assignment document, obtained from
             Naive_Bayes.py.
             
Command to run the file: python final_logistic_regression.py
Inputs            : None

Detailed Procedure: The execution starts with main function which takes train 
                    and test files as input.The call then gets forwarded to 
                    logistic regression with 30,000 steps/iterations. The 
                    learning rate is best found to be at 0.01. Training of 
                    weights is performed through logistic regression using 
                    stochastic gradient ascent. The cross entropy loss is used
                    as error metric. It also contains test code which performs
                    padding and uses the testdata, trained weight matrix to 
                    calculate sigmoid for the test set. The sigmoid result value
                    is rounded to predict negative and positive classes for the
                    records in the test set. 
'''


import pandas as pd
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json


'''
The function takes the z as the input and returns the sigmoid of z.
'''
def calculate_sigmoid(z):
    z = np.array(z,dtype=np.float64)
    g_of_z = 1 / (1 + np.exp(-z))
    return (g_of_z)

'''
Calculates the dot product of theta and X matrices to return their dot product.
'''  
def get_z(theta,X):
    return(np.dot(theta,X))

'''
The likelihood function that returns the loglikelihood. 
Was used for testing against cross entropy. But currently the program does not
 use it anywhere.
''' 
def get_likelihood(z,actual_y):
    actual_y_times_z = actual_y*z
    log_of_one_plus_e_power_z = np.log(1+np.exp(z))
    loglikelihood = np.sum(actual_y_times_z - log_of_one_plus_e_power_z)
    return(loglikelihood)

'''
Computes the cross entropy for the actual and the predicted values using the 
trained weights. Returns the cross entropy value which inludes the mean of 
individual errors.
'''
def cross_entropy_loss(data_matrix,actual_y,weight_matrix):
    clm = -(actual_y*np.log(calculate_sigmoid(np.dot(data_matrix, weight_matrix)))) - ((1-actual_y)*np.log(1-calculate_sigmoid(np.array(np.dot(data_matrix, weight_matrix),dtype=np.float64))))
    return (np.mean(clm))

'''
Computes logistic regression by using gradient ascent. Uses the input data,
weight matrix, iterations, actual labels and learning rate.
'''
def logistic_regression_with_gradient_ascent(iterations,data_matrix,weight_matrix,actual_y,alpha):
    for i in range(iterations):
        print("iteration")
        print(i)
        z = get_z(data_matrix,weight_matrix)
        h_theta_of_x = calculate_sigmoid(z)
        column_transpose =  data_matrix.T
        error = actual_y - h_theta_of_x
        gradient = np.dot(column_transpose,error)
        weight_matrix = weight_matrix+(alpha*gradient)
        
        #likelihood_value = get_likelihood(z,actual_y)
        #print(likelihood_value)
        
        cross_loss = cross_entropy_loss(data_matrix,actual_y,weight_matrix)
        #print(cross_loss)
        with open("error.txt","a+") as f:
            f.write(str(cross_loss))
            f.write("\n")
        
    return(weight_matrix)

'''
Computes logistic regression by using stochastic gradient ascent. Uses the input data,
weight matrix, iterations, actual labels and learning rate.
'''
def logistic_regression_with_stochastic_gradient_ascent(iterations,data_matrix,weight_matrix, actual_y, alpha, add_intercept = False):
    if add_intercept:
        intercept = np.ones((data_matrix.shape[0], 1))
        data_matrix = np.hstack((intercept, data_matrix))
        
    weight_matrix,cost_list=stochastic_gradient_ascent(data_matrix,actual_y,weight_matrix,alpha,iterations)
    
    cross_loss = cross_entropy_loss(data_matrix,actual_y,weight_matrix)
    print(cross_loss)
    with open("error.txt","a+") as f:
            f.write(str(cross_loss))
            f.write("\n")
            
    return weight_matrix

'''
The function embeds logic for stochastic gradient descent by using features,
actual labels, weight matrix, learning rate (alpha) and number of iterations.
'''
def stochastic_gradient_ascent(features,target,theta,learning_rate,iterations):
    m=len(target)
    cost_list=np.zeros(iterations)
    for it in range(iterations):
        cost=0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            features_i = features[rand_ind,:].reshape(1,features.shape[1])
            target_i = target[rand_ind]
            prediction=np.dot(features_i,theta)
            theta=theta+(1/m)*learning_rate*(features_i.T.dot((target_i-prediction)))
            cost+=cross_entropy_loss(features_i,target_i,theta)
    cost_list[it]=cost
    return theta,cost_list

'''
The function calculates the classification metrics such as Accuracy, precision and
F1-score. 
'''
def evaluationmetric(actual,predicted,filename):
    class_metrics = classification_report(actual.astype(int),predicted.astype(int))
    print(class_metrics)
    class_metrics_dict = classification_report(actual.astype(int),predicted.astype(int),output_dict=True)
    with open(filename,"a+") as ft:
        ft.write(json.dumps(class_metrics_dict))
            
    cm = confusion_matrix(actual.astype(int),predicted.astype(int))
    print("confusion matrix:\n",cm)
    #trained_weight_matrix = logistic_regression(50000,df_array,weight_matrix,labels_array,0.01)

'''
The main function takes the datasets as inputs, has all the important calls for
training and testing.It stores the result in the respective files. Please find 
all the
'''
def main(filename1,filename2,flag):
    
    df = pd.read_csv(filename1)
    df_test=pd.read_csv(filename2)    
    columns1 = df.columns
    columns2 = df_test.columns    
    common_columns = list(set(columns1) & set(columns2))
    df_test = df_test[common_columns]
    
    if(flag==1):
        df.label[df.label == 'positive'] = 1
        df.label[df.label == 'negative'] = 0
        df_test[df_test.label == 'positive'] = 1
        df_test.label[df_test.label == 'negative'] = 0
        
    labels_array = np.array(df.iloc[:,-1])
    df = df.iloc[:,:-1]
    df_array = np.array(df)
    
    labels_array_test = np.array(df_test.iloc[:,-1])
    df_test = df_test.iloc[:,:-1]
    df_test_array = np.array(df_test)
    
    row_count = df_array.shape[1]
    weight_matrix = np.zeros(row_count)
    trained_weight_matrix = logistic_regression_with_gradient_ascent(1000,df_array,weight_matrix,labels_array,0.01)
    
    padding = np.zeros(( df_test_array.shape[0],len(trained_weight_matrix)-df_test_array.shape[1]))
    df_test_array = np.hstack((padding, df_test_array))
    
    #testing code 
    predicted_values=np.round(calculate_sigmoid(np.dot(df_test_array,trained_weight_matrix)))
    evaluationmetric(labels_array_test,predicted_values,"results_"+filename2[:-4]+"_logistic_regression.txt")


main("stem_count_train.csv","stem_count_test.csv",1)
print("*****stem_count_train*****")

main("nostem_count_train.csv","nostem_count_test.csv",1)
print("*****nostem_count_train*****")

main("stem_binary_train.csv","stem_binary_test.csv",0)
print("*****stem_binary_train*****")

main("nostem_binary_train.csv","nostem_binary_test.csv",0)
print("*****nostem_binary_train*****")    



#Note: My system was not able to support/run large number of iterations and hence
#I have used 100 as the iteration number for  stem_count_train and nostem_count_train
#datasets. I have used 1000 for stem_binary_train and nostem_binary_train datsets.
#You can increase the number of iterations and run the algorithm to get 
#better results. 
 

        
        
        
        
    
    
    
    


    
    

