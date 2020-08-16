#!/usr/bin/env python
# coding: utf-8


'''
Author: Chaithanya Pramodh Kasula, Aishwarya Varala and Srikaran Elakurthy

Description: AIT 726 - Homework 2. The current algorithm uses a feed forward neural network for performing sentiment classification
for - no-stemming + tf-idf.

Command to run the file: python tfIdf_stem.py <path_of_file_to_tweet.zip>
For example: python tfIdf_tokenize.py "C:\Users\chait\Desktop\Spring 2020 - courses\AIT - 726\Homeworks\Homework 2 Submission\part1\homework2_part1_tokens"

Detailed Procedure: The main() function is the one that gets invoked first. It takes a paramater which is the path of the data (tweet.zip). 
The control switches to pre_processing function where the sentences extracted from the data files are preprocessed to according to the criteria mentioned in the assignment document. All the constraints for pre-processing have been satisfied. 
The data is tokenized and not stemmed*. 

The control shifts to count_vector_func_tok function where the counts for each word in sentence is computed and returned.
The function convert_to_dataframe is then used to generate a dataframe from the dictionaries that contain the word count (which are retunred by count_vector_func_tok function).
The word frequencies of positive and negative tweets are combined to generate total_word_frequency_train.csv which consists of word counts.
The function convert_to_term_frequency converts the word frequencies to term frequencies by using the np.float64(1+np.log(x)). The term frequencies of positive
and negative tweets are merged to generate total_term_frequency_train.csv. calculate_tf_idf is used to calculate the inverse document frequency for each word in each document.
The "term_frequency_inverse_document_frequency_train.csv" is generated. The same process is followed to generate "term_frequency_inverse_document_frequency_test.csv".
However, the vocabulary of train is only used for the test phase. The training() function is used to train the feed forward neural network with the training set.
The feed_forward_neural_network() function uses a fully connected neural network that uses Mean Squared Error as a loss function, Sigmoid as the activation function.
The learning rate is found to be best at 0.01. The trained model is returned and testing is performed with the test set after appropriate padding. 
The classification metrics are reported by using y_test and y_pred. The resultant metrics are written into results_part1_stem.txt. 

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
from nltk.corpus import stopwords
import math


'''
This function is used to check whether a emoji is present in the input text and return if any.
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

    sentence_count = 0
    all_sentence_word_count_dict = {}

    for sentence in vector_data_tok:
        single_sentence_word_count_dict = {}
        word_split_for_sentence =  sentence.split()
        common_tokens = list(set([value for value in word_split_for_sentence if value in vocab]))
        
        for token in common_tokens:  
            single_sentence_word_count_dict[token] = sentence.count(token)

        all_sentence_word_count_dict[sentence_count] = single_sentence_word_count_dict    
        sentence_count = sentence_count+1

    return(all_sentence_word_count_dict)

'''
The function takes the dictionary that contains the counr of each word, vocabulary, the class_flag and split_type as input.
It writes the word frequency to csv.
'''
def convert_to_dataframe(word_count_dictionary,vocab,class_flag,split_type):

    word_freq_df = pd.DataFrame(columns=vocab)
    document_names = word_count_dictionary.keys()
    
    for document in document_names:
        print(document)
        words = word_count_dictionary[document].keys()
        for word in words:
            print(word)
            word_freq_df.loc[document,word] = word_count_dictionary[document][word]
    
    word_freq_df = word_freq_df.fillna(0)
    word_freq_df.to_csv("word_frequency_"+split_type+"_"+class_flag+".csv",index=False)
    return(word_freq_df)

'''
This function is applied to every cell of the dataframe to calculate the term frequency from the word frequency csv.
'''
def tf_calculation(x):
    if(x==0):
        return(x)
    else:
        return(np.float64(1+np.log(x)))
    


'''
This function takes the split_type and input and calculates the inverse document frequency values for the dataset. 
'''
def calculate_idf(split_type):

    word_freq_df = pd.read_csv("total_word_frequency_"+split_type+".csv")
    no_of_documents = len(word_freq_df)
    column_names = word_freq_df.columns
    no_of_documents_in_which_word_appears = {}
    idf_dict = {}

    for col_name in column_names:
        print(col_name)
        no_of_documents_in_which_word_appears[col_name] = len(word_freq_df[word_freq_df[col_name]>0])
    
    #print(no_of_documents_in_which_word_appears)

    for key in no_of_documents_in_which_word_appears.keys():

        if (no_of_documents_in_which_word_appears[key]==0):
            smoothing = 1    #to handle 0 counts for words - count of word = 1 is insignificant similar to count of word = 0
                             #It also does not matter as during the multiplication of tf*idf, the zeros in the cells will be multiplied with idf values which results in a zero always
        else:
            smoothing = no_of_documents_in_which_word_appears[key]

        idf_dict[key] = np.log(no_of_documents/smoothing)

    with open("idf_dict_"+split_type+".txt","w+") as f:
        f.write(json.dumps(idf_dict))

    print(idf_dict)
    return(idf_dict)

'''
This function calculates the term frequency - inverse document frequency and writes it to a csv. It returns the tfIdf dataframe. 
'''

def calculate_tf_idf(idf_dict,split_type):

    term_freq_df = pd.read_csv("total_term_frequency_"+split_type+".csv")
    term_frequency_inverse_document_frequency = pd.DataFrame()

    for key in idf_dict.keys():
        term_frequency_inverse_document_frequency[key] = term_freq_df[key].map(lambda x: x*(idf_dict[key]))
        print("Column")
        print(key)
    
    term_frequency_inverse_document_frequency["class_label"] = ""
    term_frequency_inverse_document_frequency.loc[:1181,"class_label"] = 1
    term_frequency_inverse_document_frequency.loc[1181:,"class_label"] = 0

    term_frequency_inverse_document_frequency.to_csv("term_frequency_inverse_document_frequency_"+split_type+".csv",index=False)
    return(term_frequency_inverse_document_frequency)
    



'''
This functions takes the word frequency as input and applies tf_calculation function. It writes the values to a csv.
'''
def convert_to_term_frequency(class_flag,split_type):

    word_freq_df = pd.read_csv("word_frequency_"+split_type+"_"+class_flag+".csv")
    term_df = word_freq_df.applymap(tf_calculation)
    term_df.to_csv("term_frequency_"+split_type+"_"+class_flag+".csv",index=False)
    return(term_df)
    
    

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
The preprocessing function takes the class type and split type as inputs. All the preprocessing steps as mentioned in the assignment document are performed in the function.
'''
def pre_processing(class_type,split_type,optional_stem_vocabulary=[]):
    vocabulary = []
    bag_of_documents_with_class = []
    cachedStopWords = stopwords.words("english")

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
                line = ' '.join([word for word in line.split() if word not in cachedStopWords])
                line=BeautifulSoup(line).get_text()

                token_arr=tokenextractor(line)
                bag_of_documents.append(token_arr)
                vocabulary.append(token_arr)
        bag_of_documents_with_class.append(bag_of_documents)        

    
    tokenized_vocabulary = list(set(list(itertools.chain.from_iterable(vocabulary))))
    stemmed_vocabulary = tokenized_vocabulary


    if(split_type=="train"):
        stemmed_vocabulary_train = stemmed_vocabulary

    
    positive_set_train = [' '.join(x) for x in bag_of_documents_with_class[0]]
    negative_set_train = [' '.join(x) for x in bag_of_documents_with_class[1]]

    positive_set_train_tok_stem,pos_stem_tok = tokenize_and_stem(positive_set_train) 
    negative_set_train_tok_stem,neg_stem_tok = tokenize_and_stem(negative_set_train)

    word_count_dictionary_positive = count_vector_func_tok(positive_set_train_tok_stem,stemmed_vocabulary)
    word_count_dictionary_negative = count_vector_func_tok(negative_set_train_tok_stem,stemmed_vocabulary)

    word_freq_df_positive_train = convert_to_dataframe(word_count_dictionary_positive,stemmed_vocabulary,"positive","train")
    word_freq_df_negative_train = convert_to_dataframe(word_count_dictionary_negative,stemmed_vocabulary,"negative","train")

    total_word_frequency_df_train = pd.concat([word_freq_df_positive_train, word_freq_df_negative_train])
    total_word_frequency_df_train.to_csv("total_word_frequency_train.csv",index=False)
    
    term_frequency_df_positive_train = convert_to_term_frequency("positive","train")                                                 
    term_frequency_df_negative_train = convert_to_term_frequency("negative","train")                                                
    
    total_term_frequency_df_train = pd.concat([term_frequency_df_positive_train, term_frequency_df_negative_train])
    total_term_frequency_df_train.to_csv("total_term_frequency_train.csv",index=False)
    
    
    idf_dict = calculate_idf("train")                                                                                         

                                                         
    print("TF IDF")
    tf_idf_train = calculate_tf_idf(idf_dict,"train")                                                                                


    if(split_type=="train"):
        print("Returning from TRAIN")
        return (stemmed_vocabulary_train)


    positive_set_test = [' '.join(x) for x in bag_of_documents_with_class[0]]
    negative_set_test = [' '.join(x) for x in bag_of_documents_with_class[1]]

    positive_set_test_tok_stem,pos_stem_tok = tokenize_and_stem(positive_set_test) 
    negative_set_test_tok_stem,neg_stem_tok = tokenize_and_stem(negative_set_test)
    
    word_count_dictionary_positive = count_vector_func_tok(positive_set_test_tok_stem,optional_stem_vocabulary)
    word_count_dictionary_negative = count_vector_func_tok(negative_set_test_tok_stem,optional_stem_vocabulary)

    word_freq_df_positive_test = convert_to_dataframe(word_count_dictionary_positive,stemmed_vocabulary,"positive","test")
    word_freq_df_negative_test = convert_to_dataframe(word_count_dictionary_negative,stemmed_vocabulary,"negative","test")

    total_word_frequency_df_test = pd.concat([word_freq_df_positive_test, word_freq_df_negative_test])
    total_word_frequency_df_test.to_csv("total_word_frequency_test.csv",index=False)

    term_frequency_df_positive_test = convert_to_term_frequency("positive","test")                                               
    term_frequency_df_negative_test = convert_to_term_frequency("negative","test")                                                 

    total_term_frequency_df_train = pd.concat([term_frequency_df_positive_test, term_frequency_df_negative_test])
    total_term_frequency_df_train.to_csv("total_term_frequency_test.csv",index=False)    
    
    idf_dict_test = calculate_idf("test")

    print("TF IDF TEST")
    tf_idf_test = calculate_tf_idf(idf_dict_test,"test")   

    return(0)
    

'''
This function deals with the archiecture of Neural Network. The feed forward neural network is built with  with 2 layers with hidden vector size 20. 
The activation function is sigmoid and the learning rate is found to be best at 0.01. The loss function is Mean Squared Error. 
The number of epochs is 100 and the batch_size is 10. 
'''

def feed_forward_neural_network(X,y):

    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=len(X.columns), activation='sigmoid'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    model.fit(X, y, epochs=1, batch_size=10,verbose=True)

    return (model)

'''
This function takes the model, X_test and y_test as inputs and predicts the outputs. It calculates the metrics and returns the reults.
'''
def testing_and_metrics(model,X_test,y_test):

    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    for i in range(0,len(y_pred)):
        if(y_pred[i]>=0.5):
            y_pred[i] = 1
        else:
            y_pred[i]=0

    metrics = classification_report(y_test,y_pred,output_dict=True)
    return (metrics)

'''
The training code reads the preprocessed dataset which contains the tfIdf values. It calls the feed forward neural network model for training and 
recives the trained model. It then calls the 'testing_and_metrics()' function that tests trained model on the test and returns the metrics.
'''
def training():

    tf_idf = pd.read_csv("term_frequency_inverse_document_frequency_train.csv")
    training_columns = tf_idf.columns
    
    test_tf_idf = pd.read_csv("term_frequency_inverse_document_frequency_test.csv")
    testing_columns = test_tf_idf.columns

    common_list = [value for value in testing_columns if value in training_columns]

    test_tf_idf = pd.read_csv("term_frequency_inverse_document_frequency_test.csv", usecols = common_list)

    for i in range(len(test_tf_idf.columns),len(training_columns)):
        print(i)
        test_tf_idf.loc[:,i] = ""
        test_tf_idf.loc[:,i] = 0
    
    test_tf_idf.to_csv("term_frequency_inverse_document_frequency_test_stacked.csv",index=False)
    
    test_tf_idf = pd.read_csv("term_frequency_inverse_document_frequency_test_stacked.csv")

    X = tf_idf.iloc[:,:-1]
    y = tf_idf.iloc[:,-1]

    X_test = test_tf_idf.loc[:, test_tf_idf.columns != 'class_label']
    y_test = test_tf_idf["class_label"]

    model = feed_forward_neural_network(X,y)
    
    metrics = testing_and_metrics(model,X_test,y_test)

    return(metrics)



'''
The main function takes the input as the path of the tweet folder, invokes the preprocessing, dataset preperation and training related functions. It recieves the metrics and writes the results in to results_part1_tokens.txt file.
'''
def main(path):
    
    train_set_positive = path+"/tweet/train/positive/"
    train_set_negative = path+"/tweet/train/negative/"
    class_type = [train_set_positive,train_set_negative]
    stemmed_vocabulary_train = pre_processing(class_type,"train")

    print("IN TEST")
    test_set_positive = path+"/tweet/test/positive/"
    test_set_negative = path+"/tweet/test/negative/"
    class_type_test = [test_set_positive,test_set_negative]
    pre_processing(class_type_test,"test",stemmed_vocabulary_train)
    
    metrics = training()

    with open("results_part1_tokens.txt","w") as f:
        f.write(json.dumps(metrics))

    print(metrics)
    
    

main(sys.argv[1])
#main("/home/ckasula/NLP/homework2")




