import tensorflow as tf
#tf.random.set_seed(3)
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from keras.layers import Input,Embedding,Conv1D,Dense, MaxPooling1D,Flatten,Dropout
from keras import Model
from keras import regularizers
from keras import optimizers
from keras import metrics
from keras.utils import np_utils
from keras.backend import concatenate
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras import callbacks
import pymysql
import time
import traceback
#from keras import backend as K
#K.set_image_data_format('channels_last')

db = pymysql.connect(host="localhost", user="root", password="123456", database="Dateset")
cursor = db.cursor()
def DB_Connect(sql,flag):
    results = []
    try:
        if flag == "select":
            cursor.execute(sql)
            results = cursor.fetchall()
        if flag=="one":
            cursor.execute(sql)
            results = cursor.fetchone()
        if flag == "update":
            cursor.execute(sql)
            db.commit()
        return results
    except Exception as ex:
        print(sql)
        traceback.print_exc()


#  experiment 2 CNN_model
def TextCNN_model_4(x_train_padded_seqs,x_train_metadata, y_train, x_test_padded_seqs,
                    x_test_metadata,y_test, embedding_matrix,sent_total,metadata):
    # model structure：word embedding -convolution*3-concat-full connection-dropout-full connection
    main_input = Input(shape=(50,), dtype='float64')
    embedder = Embedding(sent_total + 1,100, input_length=50, weights=[embedding_matrix], trainable=False)
    embed = embedder(main_input)
    main_input2 = Input(shape=(1,), dtype='float64',name='input2')
    embedder2 = Embedding(9408, 9, input_length=1, weights=[metadata], trainable=False)
    embed2 = embedder2(main_input2)
    # kernel 3,4,5
    cnn1 = Conv1D(256, 3, padding='same', data_format='channels_last',strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=38)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same',data_format='channels_last', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=37)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', data_format='channels_last',strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=36)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3,embed2], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(2,activation='softmax')(drop)
    model = Model(inputs=[main_input,main_input2], outputs=main_output)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    callback=callbacks.EarlyStopping(monitor='loss',patience=2)
    one_hot_labels = np_utils.to_categorical(y_train, num_classes=2)  
    start=time.time()
    print("CNN train begin...")
    model.fit(x=[x_train_padded_seqs,x_train_metadata], y=one_hot_labels, batch_size=512,
              epochs=20,validation_split=0.1,callbacks=callback,class_weight={0:0.3740,1:0.6260})
    end=time.time()
    print("CNN train end,usage time consuming:"+str(round(end-start,2))+",CNN test begin")
    result = model.predict(x=[x_test_padded_seqs,x_test_metadata])  
    result_labels = np.argmax(result, axis=1)  
    y_predict = list(map(str, result_labels))
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(y_predict)):
        if y_test[i] == 1 and y_predict[i] == '1':
            TP += 1
        if y_test[i] == 1 and y_predict[i] == '0':
            FN += 1
        if y_test[i] == 0 and y_predict[i] == '1':
            FP = +1
        if y_test[i] == 0 and y_predict[i] == '0':
            TN += 1
    precise = round(TP / (TP + FP), 4)
    recall = round(TP / (TP + FN), 4)    
    print("F1_Score：" + str(round(2 * precise * recall / (precise + recall), 4)))
    print("accuracy：" + str(round((TP + TN) / (TP + FP + TN + FN), 4)))


#  experiment 2 CNN_model
def TextCNN_model_5(x_train_padded_seqs,x_train_metadata, y_train, x_test_padded_seqs,
                    x_test_metadata,y_test, embedding_matrix,sent_total,metadata):
    # model structure：word embedding -convolution*3-concat-full connection-dropout-full connection
    main_input = Input(shape=(50,), dtype='float64')
    embedder = Embedding(sent_total + 1, 100, input_length=50, weights=[embedding_matrix], trainable=False)
    embed = embedder(main_input)

    main_input2 = Input(shape=(1,), dtype='float64', name='input2')
    embedder2 = Embedding(7416,9 , input_length=1, weights=[metadata], trainable=False)
    embed2 = embedder2(main_input2)
    # kenel 3,4,5
    cnn1 = Conv1D(256, 3, padding='same', data_format='channels_last', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=38)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', data_format='channels_last', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=37)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', data_format='channels_last', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=36)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3, embed2], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(2, activation='softmax')(drop)
    model = Model(inputs=[main_input, main_input2], outputs=main_output)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    callback = callbacks.EarlyStopping(monitor='loss', patience=2)
    one_hot_labels = np_utils.to_categorical(y_train, num_classes=2) 
    start = time.time()
    print("CNN train begin...")
    model.fit(x=[x_train_padded_seqs, x_train_metadata], y=one_hot_labels, batch_size=32,
              epochs=20, validation_split=0.1, callbacks=callback)
    end = time.time()
    print("CNN train end,usage time consuming:" + str(round(end - start, 2)) + ",CNN test begin")
    result = model.predict(x=[x_test_padded_seqs, x_test_metadata]) 
    result_labels = np.argmax(result, axis=1)  
    y_predict = list(map(str, result_labels))
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(y_predict)):
        if y_test[i] == 1 and y_predict[i] == '1':
            TP += 1
        if y_test[i] == 1 and y_predict[i] == '0':
            FN += 1
        if y_test[i] == 0 and y_predict[i] == '1':
            FP = +1
        if y_test[i] == 0 and y_predict[i] == '0':
            TN += 1
    precise = round(TP / (TP + FP), 4)
    recall = round(TP / (TP + FN), 4)
    print("F1_score：" + str(round(2 * precise * recall / (precise + recall), 4)))
    print("accuracy：" + str(round((TP + TN) / (TP + FP + TN + FN), 4)))


def bulid_sentence_vector(str,size,w2v):
    vec=np.zeros(size).reshape(1,size)
    count=0
    for word in str.split(' '):
        try:
            vec+=w2v.wv[word].reshape(1,size)
            count+=1
        except KeyError:
            continue
    if count!=0:
        vec/=count
    return vec

#Experiment 1
def data_initial_5():
    w2v = Word2Vec.load('/dataset_all.model')
    path = "/dataset_all.txt"
    x_train_word_ids, x_test_word_ids, y_train, y_test = [], [], [], []  
    x_train_metadata_ids, x_test_metadata_ids = [],[]      #list of metadata
    size = 100
    sent_total = 0
    begin_time = time.time()
    paper_count=0
    metadata = np.zeros((9408, 9), dtype='float32')
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            sql1 = "select Aver_span,Aver_turn ,aver_Author_num,reference_num,acd_age_aver,aver_h_index,Journal_Impact,papers,type,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            results = DB_Connect(sql1, "select")
            temp = []  # embedding
            for result in results:
                for i in range(2,8):
                    temp.append(result[i])  
            metadata[paper_count]=temp
            paper_count+=1
            Ab_T = result[0] + '.' + result[1]
            sent_total += len(Ab_T.split('.'))
    stage1 = time.time()
    print("number of sentence completed：" + str(round(stage1 - begin_time, 2)) + "seconds,", end="")
    print("constructing embedding_matrix...")

    embedding_matrix = np.zeros((sent_total + 1, 100))  # w2v 100 + metadata 5

    sen_index = 0  # sentence ID
    paper_count=0  #paper_num
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            # vec = np.zeros(size).reshape(1, size)
            sql1 = "select Aver_distance,Author_num,reference_num,acd_age_aver,aver_h_index,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            str1 = result[5] + '.' + result[6]  # concat title and abstract
            str1_ls = str1.split('.')  
            temp_index = []  # sentence_Index
            #metadata=[]
            for str2 in str1_ls:
                temp = []  # embedding
                vec = bulid_sentence_vector(str2, size, w2v)
                for i in range(100):
                    temp.append(vec[0][i])  
                embedding_matrix[sen_index] = temp  # sentenct_embedding
                temp_index.append(sen_index)  # index of sentence embedding
                sen_index += 1
            if int(paper_info[1]) <= 3532:  # HCP
                if int(paper_info[0]) % 5 == 2:  # test
                    x_test_word_ids.append(temp_index)
                    y_test.append(1)
                    x_test_metadata_ids.append([paper_count])
                else:  # train
                    x_train_word_ids.append(temp_index)
                    y_train.append(1)
                    x_train_metadata_ids.append([paper_count])

            else:  # non_HCP
                if int(paper_info[0]) % 5 == 2:
                    x_test_word_ids.append(temp_index)
                    y_test.append(0)
                    x_test_metadata_ids.append([paper_count])
                else:
                    x_train_word_ids.append(temp_index)
                    y_train.append(0)
                    x_train_metadata_ids.append([paper_count])
            paper_count+=1

    stage2 = time.time()
    print("embedding_matrix completed，+" + str(round(stage2 - stage1, 2)) + "seconds，begin CNN")
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50)  
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)
    x_train_metadata=pad_sequences(x_train_metadata_ids,maxlen=1)
    x_test_metadata = pad_sequences(x_test_metadata_ids, maxlen=1)

    TextCNN_model_4(x_train_padded_seqs,x_train_metadata, y_train, x_test_padded_seqs,
                    x_test_metadata,y_test, embedding_matrix, sent_total,metadata)

#Experiment 2
def data_initial_6():
    w2v = Word2Vec.load('/dataset11-13.model')
    path = "/11-13_train_all.txt"
    path1 = "/2015_test_all.txt"
    x_train_word_ids, x_test_word_ids, y_train, y_test = [], [], [], []  
    x_train_metadata_ids, x_test_metadata_ids = [], []  # list of metadata
    size = 100
    sent_total = 0
    begin_time = time.time()
    paper_count = 0
    metadata = np.zeros((7416, 9), dtype='float32')
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            sql1 = "select Aver_span,Aver_turn ,aver_Author_num,reference_num,acd_age_aver,aver_h_index,Journal_Impact,papers,type,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            results = DB_Connect(sql1, "select")
            temp = []  # embedding
            for result in results:
                for i in range(2, 8):
                    temp.append(result[i])  
            metadata[paper_count] = temp  # dic  of metadata
            paper_count += 1
            Ab_T = result[0] + '.' + result[1]
            sent_total += len(Ab_T.split('.'))
    with open(path1, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            sql1 = "select Aver_span,Aver_turn ,aver_Author_num,reference_num,acd_age_aver,aver_h_index,Journal_Impact,papers,type,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            results = DB_Connect(sql1, "select")
            temp = []  # embedding
            for result in results:
                for i in range(2, 8):
                    temp.append(result[i])  
            metadata[paper_count] = temp # dic  of metadata
            paper_count += 1
            Ab_T = result[0] + '.' + result[1]
            sent_total += len(Ab_T.split('.'))
    stage1 = time.time()
    print("number of sentence completed：" + str(round(stage1 - begin_time, 2)) + "seconds,", end="")
    print("constructing embedding_matrix...")
    
    embedding_matrix = np.zeros((sent_total + 1, 100))  # w2v 100 + metadata 5
    sen_index = 0  # sentence ID
    paper_count = 0  
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            # vec = np.zeros(size).reshape(1, size)
            sql1 = "select Aver_span,Aver_turn ,aver_Author_num,reference_num,acd_age_aver,aver_h_index,Journal_Impact,papers,type,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            str1 = result[6] + '.' + result[7]  # concat title and abstract
            str1_ls = str1.split('.')  
            temp_index = []  # sentenct Index
            for str2 in str1_ls:
                temp = []  # embedding
                vec = bulid_sentence_vector(str2, size, w2v)
                for i in range(100):
                    temp.append(vec[0][i])  
                embedding_matrix[sen_index] = temp  # sentenct_embedding
                temp_index.append(sen_index)  #
                sen_index += 1
            if int(paper_info[1]) <= 2145:  # HCP
                x_train_word_ids.append(temp_index)  # train
                y_train.append(1)
            else:  # non_HCP
                x_train_word_ids.append(temp_index)
                y_train.append(0)
            x_train_metadata_ids.append([paper_count])
            paper_count+=1
    with open(path1, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            # vec = np.zeros(size).reshape(1, size)
            sql1 = "select Aver_distance,Author_num,reference_num,acd_age_aver,aver_h_index,Journal_Impact,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            str1 = result[6] + '.' + result[7]  
            str1_ls = str1.split('.')  
            temp_index = []  
            for str2 in str1_ls:
                temp = []  # embedding
                vec = bulid_sentence_vector(str2, size, w2v)
                for i in range(100):
                    temp.append(vec[0][i])  
                embedding_matrix[sen_index] = temp  
                temp_index.append(sen_index) 
                sen_index += 1
            if int(paper_info[1]) <= 659:  
                # test
                x_test_word_ids.append(temp_index)
                y_test.append(1)
            else:  
                x_test_word_ids.append(temp_index)
                y_test.append(0)
            x_test_metadata_ids.append([paper_count])
            paper_count+=1
    
    stage2 = time.time()
    print("embedding_matrix completed：+" + str(round(stage2 - stage1, 2)) + "seconds，begin CNN")
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50)  
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)
    x_train_metadata = pad_sequences(x_train_metadata_ids, maxlen=1)
    x_test_metadata = pad_sequences(x_test_metadata_ids, maxlen=1)

    TextCNN_model_5(x_train_padded_seqs,x_train_metadata, y_train, x_test_padded_seqs,
                    x_test_metadata,y_test, embedding_matrix, sent_total,metadata)

if __name__=='__main__':

    data_initial_5()
    #data_initial_6()




