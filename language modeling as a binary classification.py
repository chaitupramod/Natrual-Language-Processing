#!/usr/bin/env python
# coding: utf-8

# In[ ]:

'''
Author: Chaithanya Pramodh Kasula, Aishwarya Varala and Srikaran Elakurthy

Description: AIT 726 - Homework 2.  We will perform language modeling as a binary classification of positive and negative n-grams, for n=2 in this task.

Command to run the file: python tfIdf_stem.py <path_of_file_to_tweet.zip>
For example: python tfIdf_stem.py "C:\Users\chait\Desktop\Spring 2020 - courses\AIT - 726\Homeworks\Homework 2 Submission\part2"

Detailed Procedure: The main function is the first function that gets invoked. The control then gets transferred to pre_processing() function which 
performs the pre-processing steps as mentioned in the homework document. Only the train/positive data is used for this purpose.
The values in the dataframe are one hot encoded in the prepare_train_set_one_hot() function.In this prepare_train_set_one_hot() function, one hot encoded vector is prepared for each bigram in the dataframe by using the Label Encoder and one hot encooder functions. 
The one hot encoded dataframe, the one hot encoder and label encoder are returned. These objects will later be used to perform one hot encoding for the test set.
The training function is called to train the classifier (feed forward neural network) on one-hot encoded values. The same process is repeated for test set except for the training() function call. The values in the test set are predicted by using trained model.
The results are calculated and written into a file named ngrams_result.txt.    
'''

import nltk 
import os
import re
import sys
from nltk.tokenize import word_tokenize
import itertools
from nltk.tokenize import TweetTokenizer
emoticon_tokenizer = TweetTokenizer()
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import Binarizer
from bs4 import BeautifulSoup
import emoji
from nltk import FreqDist
snowball_stemmer = nltk.stem.SnowballStemmer('english')
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
import sklearn.feature_extraction.text
from nltk.tokenize import TreebankWordTokenizer
from random import sample
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

'''
This function is used to check whether a emji is present in the input text and return if any.
'''
def text_has_emoji(text):
    i=0
    new_str=''
    arr=[]
    for character in text:
        if character in emoji.UNICODE_EMOJI:
             arr.append(character)
        else:
            new_str = new_str + text[i]
        i=i+1
    return arr,new_str
basic_regex = r'\b([A-Z]{2,})s?\b'

'''
This function finds acronyms/words with all capitals and retains them.
'''

def find_acronyms(text):
    acronyms = []

    for m in re.finditer(basic_regex, text):
        acronym = m.group(1)
        acronyms.append(acronym)

    return acronyms

'''
This function is user defined that contains all the logic required to tokenize sentences according to the mentioned criteria.
'''

def tokenextractor(text):
    emjarr,new_text=text_has_emoji(text)
    acrymnarr=find_acronyms(new_text)
    p = re.compile(basic_regex)
    cleaned_text=p.sub( '', new_text)
    tokens=word_tokenize(cleaned_text.lower())
    token_arr=tokens+acrymnarr+emjarr
    return token_arr

'''
This function is used to extract tokens conforming to all the rules mentioned in the assignment. 
'''
def tokenize_only(sentence_set):
    token_list=[]
    for i in range(0,len(sentence_set)):
        token_list.append(tokenextractor(sentence_set[i]))
    return(token_list)


'''
Used to generate the count vector for each token. 
'''
def count_vector_func_tok(vector_data_tok,vocab):    
    count_vector_tok = CountVectorizer(vocabulary=vocab,tokenizer = emoticon_tokenizer.tokenize)
    count_vector_matrix = count_vector_tok.fit_transform(vector_data_tok)
    count_vect_df_tok = pd.DataFrame(count_vector_matrix.todense(), columns=count_vector_tok.get_feature_names())
    return(count_vect_df_tok)

'''
The below function tokenizes and stems a given sentence and returns the stemmed tokoens and setences combining the stemmed tokens.
'''
def tokenize_and_stem(sentence_set):
    return_set = []
    stemmed_tokens=[]
    for i in range(0,len(sentence_set)):
        tokens = tokenextractor(sentence_set[i])
        stems = []
        for j in tokens:
            stems.append(snowball_stemmer.stem(j))
            stemmed_tokens.append(snowball_stemmer.stem(j))
        return_set.append(' '.join(stems))
    return(return_set,stemmed_tokens)

'''
This function is used to sample the second word such that the second word can be any word in the corpus except for the first word itself. 
'''
def get_sample(tokenized_vocabulary,bigrams):
    x = sample(tokenized_vocabulary, 1)[0]
    if(bigrams != x):
        return(x)
    else:
        return(get_sample(tokenized_vocabulary,bigrams))


'''
The preprocessing function takes the class type and split type as inputs. All the preprocessing steps as mentioned in the assignment document are performed in the function.
We create positive n-gram samples by collecting all pairs of adjacent tokens. For every positive sample, we create 2 negative samples by keeping the first word the same as the positive sample,
but randomly sampling the rest of the corpus for the second word. The second word is checked not to be same as the first word. A dataframe of such samples s created and returned.
'''
def pre_processing(class_type,split_type):
    vocabulary = []
    bag_of_documents_with_class = []

    for type_label in class_type:
        bag_of_documents = []
        dir_name = sorted(os.listdir(type_label))
        for file in dir_name:
            file_contents1 = open(type_label + file, 'r', encoding="utf8")
            file_contents1 = list(file_contents1)
            for g in range(0,len(file_contents1)):
                file_contents1[g] = re.sub('\n',' ', file_contents1[g])
            file_contents1 = ' '.join(file_contents1)
            file_contents = [file_contents1]
            for line in file_contents:
                #line = re.sub('\W+',' ', line) #remove all special caharacters and replace them with a space
                line = re.sub('@',' ', line) #replace @ with ' '
                line = re.sub('/',' ', line) #replace / with ' '
                line = re.sub('\n',' ', line) #replace \n with ' '
                line = re.sub('\d', '', line) #replace digits with a space
                line=BeautifulSoup(line).get_text()

                token_arr=tokenextractor(line)
                bag_of_documents.append(token_arr)
                vocabulary.append(token_arr)
        bag_of_documents_with_class.append(bag_of_documents)        


    
    tokenized_vocabulary = list(set(list(itertools.chain.from_iterable(vocabulary))))

    positive_set_train = [' '.join(x) for x in bag_of_documents_with_class[0]]

    positive_set_train_tok = tokenize_only(positive_set_train)
    positive_set_train_tok_single_list = list((list(itertools.chain.from_iterable(positive_set_train_tok))))

    tokenized_vocabulary = positive_set_train_tok_single_list 
    tokenized_vocabulary = list(set(tokenized_vocabulary))

    ngram_size=2
    df = pd.DataFrame(columns=['bigram','class_label'])

    df_index = 0

    for line in positive_set_train:
        vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,ngram_size),tokenizer=TreebankWordTokenizer().tokenize)
        vect.fit([line])
        print(line)
        bigrams_list = vect.get_feature_names()

        for bigrams in bigrams_list:
            print(bigrams)
            df.loc[df_index,"bigram"] = bigrams
            df.loc[df_index,"class_label"] = 1
            df_index = df_index+1
            df.loc[df_index,"bigram"] = bigrams[0]+" "+get_sample(tokenized_vocabulary,bigrams[0])
            df.loc[df_index,"class_label"] = 0
            df_index = df_index+1
            df.loc[df_index,"bigram"] = bigrams[0]+" "+get_sample(tokenized_vocabulary,bigrams[0])
            df.loc[df_index,"class_label"] = 0
            df_index = df_index+1

        print(df)
    
    return(df)


'''
In the below function, one hot encoded vector is prepared for each bigram in the dataframe by using the Label Encoder and one hot encooder functions. 
The one hot encoded dataframe, the one hot encoder and label encoder are returned. These objects will later be used to perform one hot encoding for the test set.
'''
def  prepare_train_set_one_hot(df):

    bigram_df_list = list(df.loc[:,"bigram"])
    values = array(bigram_df_list)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    one_hot_df = pd.DataFrame(onehot_encoded)
    one_hot_df.columns = list(label_encoder.classes_)
    one_hot_df["class_label"] = df["class_label"]
    return (one_hot_df,label_encoder,onehot_encoder)
    

'''
This function deals with the archiecture of Neural Network. The feed forward neural network is built with  with 2 layers with hidden vector size 20. 
The activation function is sigmoid and the learning rate is found to be best at 0.01. The loss function is Mean Squared Error. The metric is Accuracy.
The number of epochs is 100 and the batch_size is 32. 
'''
def feed_forward_neural_network(X,y):

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=len(X.columns), activation='sigmoid'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(X, y, epochs=20, batch_size=32,verbose=True)
    return (model)

'''
This function performs training by calling the feed_forward_neural_network() function for training the model. The return model is returned by the function.
'''
def training(df):
    training_columns = df.columns
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    model = feed_forward_neural_network(X,y)
    return(model)



'''
The main function takes the tweet/positive path. It calls the pre_processing() function that returns a dataframe. The values in the dataframe
are one hot encoded in the prepare_train_set_one_hot() function. The training function is called to train the classifier on one-hot encoded values.
The same process is repeated for test set except for the training() function call. The values in the test set are predicted by using trained model.
The results are calculated and written into a file named ngrams_result.txt. 
'''    
def main(path):
    
    train_set_positive = path+"/tweet/train/positive/"
    class_type = [train_set_positive]
    bigram_class_df = pre_processing(class_type,"train")
    one_hot_df_train,label_encoder,onehot_encoder = prepare_train_set_one_hot(bigram_class_df)
    trained_model = training(one_hot_df_train)
    test_set_positive = path+"/tweet/test/positive/"
    class_type_test = [test_set_positive]
    bigram_class_df_test = pre_processing(class_type,"test")
    test_df = bigram_class_df_test
    test_bigrams = list(test_df["bigram"])
    train_df = bigram_class_df
    train_bigrams = list(train_df["bigram"])
    common_list = [i for i in test_bigrams if i in train_bigrams]
    test_df = test_df[test_df['bigram'].isin(common_list)] 
    bigram_df_list_test = list(test_df.loc[:,"bigram"])
    values = array(bigram_df_list_test)
    integer_encoded_test = label_encoder.transform(values)
    integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
    onehot_encoded_test = onehot_encoder.transform(integer_encoded_test)
    one_hot_df_test = pd.DataFrame(onehot_encoded_test)
    one_hot_df_test["class_label"] = test_df["class_label"]
    one_hot_df_test.fillna(0,inplace=True)
    X_test = one_hot_df_test.iloc[:,:-1]
    y_test = one_hot_df_test["class_label"]
    y_predicted = trained_model.predict(X_test)

    for i in range(0,len(y_predicted)):
        if(y_predicted[i]>=0.5):
            y_predicted[i] = 1
        else:
            y_predicted[i] = 0
    
    from sklearn.metrics import classification_report
    metrics = classification_report(y_test, y_predicted,output_dict=True)
    print(metrics)
    with open("ngrams_result.txt","w+") as f:
        f.write(json.dumps(metrics))



main(sys.argv[1])
#main("/home/ckasula/NLP/ngrams_homework")

