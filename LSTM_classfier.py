import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(4)
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Sequential
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from keras import Model
from keras import metrics
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import pymysql
import time
import traceback

db = pymysql.connect(host="localhost", user="root", password="123456", database="Dataset")
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

#Experiment 1 LSTM_model
def TextLSTM_model_2(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix,sent_total):

    main_input = Input(shape=(50,), dtype='float64')
    embedder = Embedding(sent_total + 1,109 , input_length=50, weights=[embedding_matrix], trainable=False)
    embed = embedder(main_input)
    layer=LSTM(128)(embed)
    layer=Dense(128,activation="relu",name="FC1")(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(2, activation="softmax", name="FC2")(layer)
    model = Model(inputs=main_input, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    one_hot_labels = np_utils.to_categorical(y_train, num_classes=2)  
    start=time.time()
    print("LSTM train begin...")
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=512, epochs=20,validation_split=0.1,class_weight={0:0.3740,1:0.6260})
    end=time.time()
    print("LSTM train end,usage time consuming:"+str(round(end-start,2))+",LSTM test begin")
    result = model.predict(x_test_padded_seqs)  
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

#Experiment 2 LSTM_model
def TextLSTM_model_3(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix,sent_total):

    main_input = Input(shape=(50,), dtype='float64')
    embedder = Embedding(sent_total + 1,109 , input_length=50, weights=[embedding_matrix], trainable=False)
    embed = embedder(main_input)
    layer=LSTM(128)(embed)
    layer=Dense(128,activation="relu",name="FC1")(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(2, activation="softmax", name="FC2")(layer)
    model = Model(inputs=main_input, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    one_hot_labels = np_utils.to_categorical(y_train, num_classes=2)  
    start=time.time()
    print("LSTM train begin...")
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=32, epochs=20,validation_split=0.1)
    end=time.time()
    print("LSTM train end,usage time consuming:"+str(round(end-start,2))+",LSTM test begin")
    result = model.predict(x_test_padded_seqs)  
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
def data_initial_2():
    w2v = Word2Vec.load('/dataset_all.model')  # import word2vec model
    path = "/dataset_all.txt"
    x_train_word_ids, x_test_word_ids, y_train, y_test = [], [], [], [] 
    size = 100
    sent_total = 0
    begin_time = time.time()
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
           sql1 = "select Aver_distance,Author_num,reference_num,acd_age_aver,aver_h_index,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            Ab_T = result[0] + '.' + result[1]
            sent_total += len(Ab_T.split('.'))
    stage1 = time.time()
    print("number of sentence completed：" + str(round(stage1 - begin_time, 2)) + "seconds,", end="")
    print("constructing embedding_matrix...")

    embedding_matrix = np.zeros((sent_total + 1, 109))  # w2v 100 + metadata 9
    sen_index = 0  # sentence ID
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            sql1 = "select Aver_distance,Author_num,reference_num,acd_age_aver,aver_h_index,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            str1 = result[6] + '.' + result[7]  #  concat title and abstract
            str1_ls = str1.split('.')  # 
            temp_index = []  # sentence_Index

            for str2 in str1_ls:
                temp = []  # embedding
                for i in range(6):
                    temp.append(result[i])  
                # temp.append(result[0])
                vec = bulid_sentence_vector(str2, size, w2v)
                for i in range(100):
                    temp.append(vec[0][i])   
                embedding_matrix[sen_index] = temp  # sentence embedding
                temp_index.append(sen_index)  # index of sentence embedding
                sen_index += 1
            if int(paper_info[1]) <= 3532:  # HCP
                if int(paper_info[0]) % 5 == 2:  # test
                    x_test_word_ids.append(temp_index)
                    y_test.append(1)
                else:  # train
                    x_train_word_ids.append(temp_index)
                    y_train.append(1)
            else:  # non_HCP
                if int(paper_info[0]) % 5 == 2:
                    x_test_word_ids.append(temp_index)
                    y_test.append(0)
                else:
                    x_train_word_ids.append(temp_index)
                    y_train.append(0)

    stage2 = time.time()
   print("embedding_matrix completed：+" + str(round(stage2 - stage1, 2)) + "seconds，begin LSTM")
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50)  
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)
    TextLSTM_model_2(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix, sent_total)

#Experiment 2
def data_initial_3():
    w2v = Word2Vec.load('/dataset_all.model')
    path = "/11-13_train_all.txt"
    path1 = "/2015_test_all.txt"
    x_train_word_ids, x_test_word_ids, y_train, y_test = [], [], [], [] 
    size = 100
    sent_total = 0
    begin_time = time.time()
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            sql1 = "select Abstract,Title from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            Ab_T = result[0] + '.' + result[1]
            sent_total += len(Ab_T.split('.'))
    with open(path1, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            sql1 = "select Abstract,Title from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            Ab_T = result[0] + '.' + result[1]
            sent_total += len(Ab_T.split('.'))
    stage1 = time.time()
    print("number of sentence completed：" + str(round(stage1 - begin_time, 2)) + "seconds,", end="")
    print("constructing embedding_matrix...")

    embedding_matrix = np.zeros((sent_total + 1, 109))  # w2v 100 + metadata 5
    sen_index = 0  # sentence ID
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
                for i in range(6):
                    temp.append(result[i])  
                # temp.append(result[0])
                vec = bulid_sentence_vector(str2, size, w2v)
                for i in range(100):
                    temp.append(vec[0][i])  
                embedding_matrix[sen_index] = temp  #  sentenct_embedding
                temp_index.append(sen_index)  # 
                sen_index += 1
            if int(paper_info[1]) <=2145: # HCP
                x_train_word_ids.append(temp_index)  # train
                y_train.append(1)
            else:  # 非高被引标签为0
                x_train_word_ids.append(temp_index)
                y_train.append(0)
    with open(path1, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            # vec = np.zeros(size).reshape(1, size)
            sql1 ="select Aver_distance,Author_num,reference_num,acd_age_aver,aver_h_index,Journal_Impact,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            result = DB_Connect(sql1, "one")
            str1 = result[6] + '.' + result[7]  
            str1_ls = str1.split('.')  
            temp_index = []  
            for str2 in str1_ls:
                temp = []  # embedding值
                for i in range(6):
                    temp.append(result[i])  
                vec = bulid_sentence_vector(str2, size, w2v)
                for i in range(100):
                    temp.append(vec[0][i])  
                embedding_matrix[sen_index] = temp  
                temp_index.append(sen_index)  
                sen_index += 1
            if int(paper_info[1]) <=659: # HCP
                  # test
                x_test_word_ids.append(temp_index)
                y_test.append(1)
            else:  
                x_test_word_ids.append(temp_index)
                y_test.append(0)

    stage2 = time.time()
   print("embedding_matrix completed：+" + str(round(stage2 - stage1, 2)) + "seconds，begin LSTM")
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50)  
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)
    TextLSTM_model_3(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix, sent_total)

if __name__=='__main__':
    data_initial_2()
    #data_initial_3()
