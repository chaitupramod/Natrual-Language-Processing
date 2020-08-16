'''
Author: Chaithanya Pramodh Kasula, Aishwarya Varala and Srikaran Elakurthy

Description: AIT 726 - Homework 3. The current algorithm performs entity based recognition with different architectures.


Command to run the file: python entityRecognition.py
Requirement: The data set needs to be in the same folder as this file and the required packages are to be installed.

Detailed Procedure: The main() function is the one that gets invoked first. It reads the train.txt file and class the pre_processing function to pre process the data according to the condition stated in the assignment. The sentences in text are converted to sequences in the tokenize_and_encode_words function. The same is performed with the entities. The pretrained word2vec embeddings are loaded and the sequences are trained to learn the entities. Different architectures are bulit and fitted to the data. The validation data is also made to go through the same preprocessing steps and are provided to the models to perform validation. The trained models are tested by passing the test.txt. The results are checked by invoking the get_results function in the conlleval.py.
'''



import nltk
import pandas as pd
import sys
import re
import pickle
import random
import numpy as np                                                                                      
from itertools import chain
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim import models
from keras.layers import Dense, Input, LSTM, Embedding, Bidirectional, Dropout, SimpleRNN,GRU
from keras.models import Model
from keras.initializers import Constant


'''
This function reads the file by its name and returns the list of lines in a file.
'''
def read_file(filename):
    with open(filename,"r") as f:
        return(f.readlines())


'''
This function takes the list of lines read from the read_file function and preprocesses the lines as stated in the homework assignment document.
It returns the preprocessed sentences, list of words and their entities. The function also performs padding as stated in the assignment document.
'''
def pre_processing(lines_list):

    lines_list = [i.replace("\n",'') for i in lines_list]
    lines_list = [i.replace("-DOCSTART- -X- -X- O",'') for i in lines_list]
    lines_list = list(filter(None, lines_list))

    for i in range(0,len(lines_list)):
        line_split = lines_list[i].split(" ")
        lines_list[i] = line_split[0]+" "+line_split[3]

    lines_list = ' '.join(lines_list)
    sep = " . O"
    lines_list_in_sentences = lines_list.split(sep)
    lines_list_in_sentences = list(filter(None, lines_list_in_sentences))
    lines_list_in_sentences = [x.strip()+sep for x in lines_list_in_sentences]


    for num,line in enumerate(lines_list_in_sentences):
        acronym = ' '.join(list(filter(None, [x.strip() for x in re.findall(r"\b[A-Z\s\-]+\b", line)])))
        acronym = acronym.split(" ")
        new_line = []
        words = line.split(" ")

        for word in words:
            if word in acronym:
                new_line.append(word)
            else:
                new_line.append(word.lower())

        lines_list_in_sentences[num] = new_line
    

    lines_list_in_sentences = lines_list_in_sentences[:1000]  #taking only 1000 sentences out of 7361 sentences
    
    words_list = [i[0::2] for i in lines_list_in_sentences]
    entity_list = [i[1::2] for i in lines_list_in_sentences]
    return(lines_list_in_sentences,words_list,entity_list)

 
'''
This function takes the input the vocabulary count, the list of sentences and the a user defined max length which limits the number of words in a sentence. The tokenizer is built and is fit to the text sentences. The word_index dictionary is built. The sentences are converted to sequences using the test_to_sequences function of Keras. The pad_sequences function is used to pad the sentence with zeroes to match the highest length of sentence in the dataset.
'''
def tokenize_and_encode_words(vocab_count, words_list_train, MAX_LEN):
    
    tokenizer = Tokenizer(num_words=vocab_count,filters='')
    words_list_sentences_temp = []

    for sentence in words_list_train:
        words_list_sentences_temp.append(' '.join(sentence))

    tokenizer.fit_on_texts(words_list_sentences_temp)
    word_index_dict = tokenizer.word_index
    word_index_dict['<pad>'] = 0
    print('Unique tokens Count')
    print(len(word_index_dict.keys()))
    mapped_sentences_train = tokenizer.texts_to_sequences(words_list_sentences_temp)
    mapped_sentences_train_padded = pad_sequences(mapped_sentences_train, maxlen=MAX_LEN)
    return(mapped_sentences_train_padded, word_index_dict, tokenizer)

'''
The function takes the entity vocabulary, list of entities and the MAX_LEN as inputs. It builds a tokenizer which is later used to fit the entities by using the fit_on_texts function of Keras. The entities_index dictionary is built. The text_to_sequences is applied to convert entities to sequences/numbers. The entities are padded with zeroes.
'''
def tokenize_and_encode_entities(entities_vocab, entity_list_train, MAX_LEN):

    entities_tokenizer = Tokenizer(num_words=len(entities_vocab),filters='', lower=False)

    entities_list_sentences_temp = []

    for entities in entity_list_train:
        entities_list_sentences_temp.append(' '.join(entities))

    entities_tokenizer.fit_on_texts(entities_list_sentences_temp)
    entities_index = entities_tokenizer.word_index
    entities_index['<pad>'] = 0

    index_entities = dict((v, k) for k, v in entities_index.items())
    mapping_predictions_to_entities_dict = index_entities
    mapping_predictions_to_entities_dict[entities_index['<pad>']] = '0'
    print('Unique entities count')
    print(len(entities_index))

    mapped_entities_train = entities_tokenizer.texts_to_sequences(entities_list_sentences_temp)

    mapped_entities_train_padded = pad_sequences(mapped_entities_train, maxlen=MAX_LEN)

    return(mapped_entities_train_padded, entities_index, entities_tokenizer, mapping_predictions_to_entities_dict)


'''
This function loads the word2vec model and returns the loaded model.
'''
def get_word2vec_model():
    
    model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    return(model)
    

'''
The embedding matrix which is to be used in the embedding layer is built inthis function by extracting the vector for the word in the vocabulary from the word2vec embeddings. For a word in the vocabulary, if the relevant embedding is not found, then certain modifications such as capitalizing the whole word, capitalizing the first word, lower casing the whole word would be tried. Even after that if the embedding is not found, then  a 300 dimensional vector with random numbers is assigned for that particular token/word.
'''
def prepare_embedding_matrix(vocab_count, word_index, model):

    num_words = min(vocab_count, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, 300))

    print("Shape of Embedding Matrix")
    print(embedding_matrix.shape)

    for word, i in word_index.items():

        if i >= vocab_count:
            continue
        
        try:
            embedding_vector = model[word]
        except:
            try:
                embedding_vector = model[word.title()]
            except:
                try:
                    embedding_vector = model[word.upper()]
                except:
                     embedding_vector = np.array([round(np.random.rand(),8) for i in range(0,300)])

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return(embedding_matrix)

'''
This function is used to build the architecture of the model. It imports the embedding layer and the relevant models from keras. The dropout function is used for regularization. All the other specifications have been provided as specified in the assignment document.
'''
def train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index,model_type,model_save_name):

    #----------------------------- bidirectional LSTM TRAINABLE---------------------------------------#
    
    if(model_type == "simple_RNN_trainable"):
        ifTrainable=True
    else:
        ifTrainable=False


    embedding_layer = Embedding(len(word_index)+1, 
                                            300, 
                                            embeddings_initializer=Constant(embedding_matrix),
                                            input_length=MAX_LEN,
                                            trainable=ifTrainable)

    sentence_input = Input(shape=(MAX_LEN,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)

    if(model_type=="bi_directional_lstm"):
        x = Bidirectional(LSTM(256, return_sequences=True))(embedded_sequences)

    elif(model_type=="lstm"):
        x = (LSTM(256, return_sequences=True))(embedded_sequences)

    elif(model_type=="bi_directional_GRU"):
        x = Bidirectional(GRU(256, return_sequences=True))(embedded_sequences)

    elif(model_type=="GRU"):
        x = (GRU(256, return_sequences=True))(embedded_sequences)

    elif(model_type=="simple_RNN" or "simple_RNN_trainable"):
        x = (SimpleRNN(256, return_sequences=True))(embedded_sequences)

    elif(model_type=="bi_directional_RNN"):
        x = Bidirectional(SimpleRNN(256, return_sequences=True))(embedded_sequences)


    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(len(entities_index), activation='softmax')(x)

    model = Model(sentence_input, predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['sparse_categorical_accuracy'])

    print(model.summary())
    model.save(model_save_name+"_256_softmax.h5")
    return(model, model_save_name)



'''
It takes the model name and loades the model from the saved file. It predicts the test set sentences and returns predictions.
'''
def predict_entities(model_name,test_sequences_padded):
    from keras.models import load_model

    model = load_model(model_name)
    test_predictions = model.predict(test_sequences_padded)
    return(test_predictions)


'''
This function is used to convert the list of list containig probabilites predicted by the softmax function to the relevant entity in word format. 
'''
def get_predictions_in_words(test_predictions, words_list_test,mapping_predictions_to_entities_dict):

    predicted_tags = []
    trimmed_sentences = []

    for sentence, sentence_pred in zip(words_list_test, test_predictions):
        print(sentence)
        print(len(sentence_pred))
        if(len(sentence)>100):
            print("*"*100)
            sentence=sentence[:100]

        pred_tags = np.argmax(sentence_pred, axis=1)
        
        pred_tags = map(mapping_predictions_to_entities_dict.get,pred_tags)

        pred_tags = list(pred_tags)[-len(sentence):]

        if(len(sentence)!=len(pred_tags)):
            print("IN BREAK")
            break

        trimmed_sentences.append(sentence)        
        predicted_tags.append(pred_tags)

    return(predicted_tags,trimmed_sentences)


'''
The main function reads the train.txt, valid.txt and test.txt functions and calls the relevant functions accordingly. It also consists the model fitting code followed by validation and test codes. The results are also written into a file.
'''
def main():

    
    train_lines = read_file("train.txt")
    lines_list_in_sentences_train,words_list_train,entity_list_train= pre_processing(train_lines)
    print("RETURNED COUNT")
    print(len(words_list_train))
    print(len(entity_list_train))

    vocabulary_train = list(set(chain(*words_list_train)))
    entities_vocab = list(set(chain(*entity_list_train)))

    vocab_count = len(vocabulary_train)
    print("vocab_count")
    print(vocab_count)

    MAX_LEN=100

    mapped_sentences_train_padded, word_index_dict, tokenizer = tokenize_and_encode_words(vocab_count, words_list_train, MAX_LEN)
    
    mapped_entities_train_padded, entities_index, entities_tokenizer, mapping_predictions_to_entities_dict = tokenize_and_encode_entities(entities_vocab, entity_list_train, MAX_LEN)

    mapped_entities_train_padded = np.expand_dims(mapped_entities_train_padded, -1)
    
    validation_lines = read_file("valid.txt")
    lines_list_in_sentences_valid,words_list_valid,entity_list_valid = pre_processing(validation_lines)
    
    words_list_sentences_valid_temp = []

    for sentence in words_list_valid:
        words_list_sentences_valid_temp.append(' '.join(sentence))

    valid_sentences = tokenizer.texts_to_sequences(words_list_sentences_valid_temp)

    valid_sentences_padded = pad_sequences(valid_sentences, maxlen=MAX_LEN)

    valid_entities_temp = []
    for entity_list in entity_list_valid:
        valid_entities_temp.append(' '.join(entity_list))
    
    valid_entities = entities_tokenizer.texts_to_sequences(valid_entities_temp)
    valid_entities_padded = pad_sequences(valid_entities, maxlen=MAX_LEN)
    valid_entities_padded = np.expand_dims(valid_entities_padded, -1)


        
    word2vec_model = get_word2vec_model()

    embedding_matrix = prepare_embedding_matrix(vocab_count,word_index_dict, word2vec_model)


    simple_RNN_model_trainable, result_file_name = train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index_dict,"simple_RNN_trainable","simple_RNN_trainable")
    
    #bi_lstm_model, result_file_name = train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index_dict,"bi_directional_lstm","bi_directional_lstm")

    #bi_gru_model, result_file_name = train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index_dict,"bi_directional_GRU","bi_directional_GRU")

    #gru_model, result_file_name = train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index_dict,"GRU","GRU")

    #lstm_model,result_file_name = train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index_dict,"lstm","lstm")

    #rnn_model,result_file_name = train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index_dict,"simple_RNN","simple_RNN")

    #bi_rnn_model,result_file_name = train_model(vocab_count,embedding_matrix,MAX_LEN,entities_index,word_index_dict,"bi_directional_RNN","bi_directional_RNN")
    


    simple_RNN_model_trainable.fit(mapped_sentences_train_padded, mapped_entities_train_padded,batch_size=32,epochs=1,validation_data=(valid_sentences_padded,valid_entities_padded))

    #bi_lstm_model.fit(mapped_sentences_train_padded, mapped_entities_train_padded,batch_size=32,epochs=100,validation_data=(valid_sentences_padded,valid_entities_padded))
    
    #bi_gru_model.fit(mapped_sentences_train_padded, mapped_entities_train_padded,batch_size=32,epochs=100,validation_data=(valid_sentences_padded,valid_entities_padded))

    #gru_model.fit(mapped_sentences_train_padded, mapped_entities_train_padded,batch_size=32,epochs=100,validation_data=(valid_sentences_padded,valid_entities_padded))

    #lstm_model.fit(mapped_sentences_train_padded, mapped_entities_train_padded,batch_size=32,epochs=100,validation_data=(valid_sentences_padded,valid_entities_padded))

    #rnn_model.fit(mapped_sentences_train_padded, mapped_entities_train_padded,batch_size=32,epochs=100,validation_data=(valid_sentences_padded,valid_entities_padded))

    #bi_rnn_model.fit(mapped_sentences_train_padded, mapped_entities_train_padded,batch_size=32,epochs=100,validation_data=(valid_sentences_padded,valid_entities_padded))
    
    test_lines = read_file("test.txt")
    lines_list_in_sentences_test,words_list_test,entity_list_test = pre_processing(test_lines)
   
    words_list_test_temp = []
    for words_list in words_list_test:
        words_list_test_temp.append(words_list)

    test_sentences = tokenizer.texts_to_sequences(words_list_test_temp)
    test_sentences_padded = pad_sequences(test_sentences, maxlen=MAX_LEN)
    
    entity_list_test_temp = []

    for entity_list in entity_list_test:
        entity_list_test_temp.append(entity_list)

    test_entities = entities_tokenizer.texts_to_sequences(entity_list_test_temp)

    test_entities_padded = pad_sequences(test_entities, maxlen=MAX_LEN)
    test_entities_padded = np.expand_dims(test_entities_padded, -1)



    test_predictions = predict_entities("simple_RNN_trainable_256_softmax"+".h5",test_sentences_padded)
    #test_predictions = predict_entities("bi_directional_lstm_256_softmax.h5",test_sentences_padded)
    #test_predictions = predict_entities("bi_directional_GRU_256_softmax.h5",test_sentences_padded)
    #test_predictions = predict_entities("GRU_256_softmax.h5",test_sentences_padded)
    #test_predictions = predict_entities("lstm_256_softmax.h5",test_sentences_padded)
    #test_predictions = predict_entities("simple_RNN_256_softmax.h5",test_sentences_padded)
    #test_predictions = predict_entities("bi_directional_RNN_256_softmax.h5",test_sentences_padded)


    predictions_in_words,trimmed_sentences = get_predictions_in_words(test_predictions, words_list_test,mapping_predictions_to_entities_dict)


    #you can have sentences greater than 1000 for test set sentences
    
    from sklearn.metrics import classification_report
    
    
    import itertools 

    entity_list_test = [i[:100] for i in entity_list_test]

    trimmed_sentences = list(itertools.chain(*trimmed_sentences))
    predictions_in_words = list(itertools.chain(*predictions_in_words))
    entity_list_test = list(itertools.chain(*entity_list_test))

    print(classification_report(entity_list_test,predictions_in_words))

    with open(result_file_name+".txt","w") as f:
        for i in range(0,len(trimmed_sentences)):
            f.write(trimmed_sentences[i]+"  "+entity_list_test[i]+" "+predictions_in_words[i]+"\n")

    
main()

