#!/usr/bin/env python
# coding: utf-8

# In[ ]:
'''
Author: Chaithanya Pramodh Kasula, Srikaran Elakurthy, Aishwarya Varala

Description: AIT 726 - Homework 1. The current program uses Naive Bayes Classifier to train and test the tweets dataset.

Command to run the file: python Naive_Bayes.py <path_of_file_to_tweet.zip>
For example: python Naive_Bayes.py C:\Users\chait\Desktop\Spring 2020 - courses\AIT - 726\Homeworks\Homework 1

Detailed Procedure: The main() function is the one that gets invoked first. It takes a paramater which is the path of the data (tweet.zip). 
The control switches to pre_processing function where the sentences extracted from the data files are preprocessed to according to the criteria mentioned in the assignment document. All the constraints for pre-processing have been satisfied. 
The pre-procesed data and the vocabulary are returned. In the same program, it generates four datasets (termfrequency and binary) as dataframes.
The training and testing calls are made in the main function and hence NVB_train and NVB_test functions are caled. 
The results are dumped and analyse in the evaluationmetric function is used to evaluate the prectiction.
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
The preprocessing function takes the class type and split type as inputs. All the preprocessing steps as mentioned in the assignment document are performed in the function.
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

    stemmed_vocabulary = set([snowball_stemmer.stem(x) for x in tokenized_vocabulary])

    positive_set_train = [' '.join(x) for x in bag_of_documents_with_class[0]]
    negative_set_train = [' '.join(x) for x in bag_of_documents_with_class[1]]


    positive_set_train_tok = tokenize_only(positive_set_train)
    positive_set_train_tok_single_list = list((list(itertools.chain.from_iterable(positive_set_train_tok))))

    negative_set_train_tok = tokenize_only(negative_set_train)
    negative_set_train_tok_single_list = list((list(itertools.chain.from_iterable(negative_set_train_tok))))

    tokenized_vocabulary = positive_set_train_tok_single_list + negative_set_train_tok_single_list
    tokenized_vocabulary = list(set(tokenized_vocabulary))

    count_vector_tok_positive_df = count_vector_func_tok(positive_set_train,tokenized_vocabulary)
    count_vector_tok_positive_df['label'] = 'positive'
    count_vector_tok_negative_df = count_vector_func_tok(negative_set_train,tokenized_vocabulary)
    count_vector_tok_negative_df['label'] = 'negative'


    token_df = pd.concat([count_vector_tok_positive_df, count_vector_tok_negative_df])
    token_df.to_csv("nostem_count_"+split_type+".csv",index=False)

    binary_token_df = token_df

    binary_token_df.label[binary_token_df.label == 'positive'] = 1 
    binary_token_df.label[binary_token_df.label == 'negative'] = 0 
    transformer = Binarizer().fit(binary_token_df)

    binary_token_df = pd.DataFrame(transformer.transform(token_df),columns=token_df.columns)
    binary_token_df.to_csv("nostem_binary_"+split_type+".csv",index=False)


    positive_set_train_tok_stem,pos_stem_tok = tokenize_and_stem(positive_set_train) 
    negative_set_train_tok_stem,neg_stem_tok = tokenize_and_stem(negative_set_train)

    df_positive_count_vector = count_vector_func_tok(positive_set_train_tok_stem,stemmed_vocabulary)
    df_positive_count_vector['label'] = 'positive'
    df_negative_count_vector = count_vector_func_tok(negative_set_train_tok_stem,stemmed_vocabulary)
    df_negative_count_vector['label'] = 'negative'

    freq_count_df = pd.concat([df_positive_count_vector, df_negative_count_vector])
    freq_count_df.to_csv("stem_count_"+split_type+".csv",index=False)

    binary_df = freq_count_df

    binary_df.label[binary_df.label == 'positive'] = 1 
    binary_df.label[binary_df.label == 'negative'] = 0 
    transformer = Binarizer().fit(freq_count_df)

    binary_data = pd.DataFrame(transformer.transform(freq_count_df),columns=freq_count_df.columns)
    binary_data.to_csv("stem_binary_"+split_type+".csv",index=False)

    return (token_df,binary_token_df,freq_count_df,binary_data,positive_set_train_tok_single_list,negative_set_train_tok_single_list,pos_stem_tok,neg_stem_tok,len(tokenized_vocabulary),len(stemmed_vocabulary))

'''
This function is used to train the Naive Baye's classifier for the given input data. It takes the positive set of tokens, negative set of tokens, length of vocabulary and a flag determining whether the data set is binary or not. 
'''
def NVB_train(pos_set,neg_set,vocablen,is_binary):
    
    freqDist_pos = FreqDist(pos_set)
    freqDist_neg = FreqDist(neg_set)
    if(is_binary==True):
        for negw in freqDist_neg:
            prob_neg=(1+1)/(len(neg_set)+vocablen)
            freqDist_neg[negw]=prob_neg

        for posw in freqDist_pos:
            probp=(1+1)/(len(pos_set)+vocablen)
            freqDist_pos[posw]=probp
    elif(is_binary==False):
        for negw in freqDist_neg:
            prob_neg=(freqDist_neg[negw]+1)/(len(neg_set)+vocablen)
            freqDist_neg[negw]=prob_neg

        for posw in freqDist_pos:
            probp=(freqDist_pos[posw]+1)/(len(pos_set)+vocablen)
            freqDist_pos[posw]=probp
        
    return freqDist_pos,freqDist_neg


'''
The below function uses precison, recall, f1-score and Accuracy to evaluate the classification task.
'''
def evaluationmetric(actual,predicted,filename):
    class_metrics = classification_report(actual.label.astype(int),predicted.predicted_class.astype(int))
    print(class_metrics)
    class_metrics_dict = classification_report(actual.label.astype(int),predicted.predicted_class.astype(int),output_dict=True)
    with open(filename,"a+") as ft:
        ft.write(json.dumps(class_metrics_dict))
    
    cm = confusion_matrix(actual.label.astype(int),predicted.predicted_class.astype(int))
    print("confusion matrix:\n",cm)
    
'''
This function performs the testing using the trained Naive Bayes model and the input test dataset. It takes inputs: test dataset, path of datasets of each class, positive set of words, negative set of words, length of postive tokens, length of negative tokens and length of the vocabulary
'''
def NBV_test(tc_test,class_path,pos_prob_words,neg_prob_words,pos_toklen,neg_toklen,vocablen):    
    column_list_df = tc_test.columns
    column_list_df_new = list(column_list_df)
    resdf=pd.DataFrame()
    resdf['positive_probability']=''
    resdf['negative_probability']=''
    resdf['predicted_label']=''
    resdf['predicted_class']=''
    poscnt=len(os.listdir(class_path[0]))
    negcnt=len(os.listdir(class_path[1]))
    pos_doc_prob=poscnt/(poscnt+negcnt)
    neg_doc_prob=negcnt/(poscnt+negcnt)
    for i in range(0,len(tc_test)):
        row_val = np.array(tc_test.iloc[i,:])
        greater_than_zero_indexes = np.argwhere(row_val > 0)
        greater_than_zero_words = list(column_list_df[greater_than_zero_indexes])
        pos_tot_prob=1
        neg_tot_prob=1
        for word in greater_than_zero_words:

            if(pos_prob_words[word] != 0):
                pos_tot_prob=pos_tot_prob*pow(pos_prob_words[word],tc_test.iloc[i,column_list_df_new.index(word)])#remove power if needed
                #pos_tot_prob=pos_tot_prob*pos_prob_words[word]
            elif(neg_prob_words[word]!=0):
                wordprob=(0+1)/(pos_toklen+vocablen)
                pos_tot_prob=pos_tot_prob*wordprob

            if(neg_prob_words[word]!=0):
                neg_tot_prob=neg_tot_prob*pow(neg_prob_words[word],tc_test.iloc[i,column_list_df_new.index(word)])#remove power if needed
                #neg_tot_prob=neg_tot_prob*neg_prob_words[word]
            elif(pos_prob_words[word] != 0):
                wordprob=(0+1)/(neg_toklen+vocablen)
                neg_tot_prob=neg_tot_prob*wordprob

        pos_class_prob = pos_doc_prob*pos_tot_prob
        neg_class_prob = neg_doc_prob*neg_tot_prob
        if(pos_class_prob>=neg_class_prob):
            new_row = {'positive_probability':pos_class_prob, 'negative_probability':neg_class_prob, 'predicted_label':'Positive','predicted_class':1}
            resdf = resdf.append(new_row, ignore_index=True)
        else:
            new_row = {'positive_probability':pos_class_prob, 'negative_probability':neg_class_prob, 'predicted_label':'Negative','predicted_class':0}
            resdf = resdf.append(new_row,ignore_index=True)
    return resdf        


'''
This function takes the path of the tweet.zip file and genertes different datasets as specified. It also has function calls to train and test functions. 
'''    
def main(path):
    train_set_positive = path+"/tweet/train/positive/"
    train_set_negative = path+"/tweet/train/negative/"
    class_type = [train_set_positive,train_set_negative]
    tc,tb,sc,sb,pos_set_tok,neg_set_tok,stem_pos,stem_neg,len_voc_tok,len_voc_stem=pre_processing(class_type,"train")

    prob_pos_words,prob_neg_words=NVB_train(pos_set_tok,neg_set_tok,len_voc_tok,False)
    prob_pos_words_bin,prob_neg_words_bin=NVB_train(pos_set_tok,neg_set_tok,len_voc_tok,True)
    prob_pos_stemwords,prob_neg_stemwords=NVB_train(stem_pos,stem_neg,len_voc_stem,False)
    prob_pos_stemwords_bin,prob_neg_stemwords_bin=NVB_train(stem_pos,stem_neg,len_voc_stem,True)

    test_set_positive = path+"/tweet/test/positive/"
    test_set_negative = path+"/tweet/test/negative/"

    class_type_test = [test_set_positive,test_set_negative]

    tc_test,tb_test,sc_test,sb_test,pos_set_tok_test,neg_set_tok_test,stem_pos_test,stem_neg_test,len_voc_tok_test,len_voc_stem_test=pre_processing(class_type_test,"test")
    final_test_count_df=NBV_test(tc_test,class_type,prob_pos_words,prob_neg_words,len(pos_set_tok),len(neg_set_tok),len_voc_tok)
    final_test_count_df.to_csv('final_test_count_df.csv')  
    evaluationmetric(tc_test,final_test_count_df,"nostem_termFrequency_results_naive_bayes.txt")

    final_test_bin_df=NBV_test(tb_test,class_type,prob_pos_words_bin,prob_neg_words_bin,len(pos_set_tok),len(neg_set_tok),len_voc_stem)
    final_test_bin_df.to_csv('final_test_bin_df.csv') 
    evaluationmetric(tb_test,final_test_bin_df,"nostem_binary_results_naive_bayes.txt")
    
    final_test_stemcount_df=NBV_test(sc_test,class_type,prob_pos_stemwords,prob_neg_stemwords,len(stem_pos),len(stem_neg),len_voc_stem)
    final_test_stemcount_df.to_csv('final_test_stemcount_df.csv')
    evaluationmetric(sc_test,final_test_stemcount_df,"stem_termFrequency_results_naive_bayes.txt")

    final_test_stemcount_bin_df=NBV_test(sb_test,class_type,prob_pos_stemwords_bin,prob_neg_stemwords_bin,len(stem_pos),len(stem_neg),len_voc_stem)
    final_test_stemcount_bin_df.to_csv('final_test_stemcount_bin_df.csv')
    evaluationmetric(sb_test,final_test_stemcount_bin_df,"stem_binary_results_naive_bayes.txt")
    


main(sys.argv[1])
#main("C:/Users/chait/Desktop/Spring 2020 - courses/AIT - 726/Homeworks/Homework 1")

