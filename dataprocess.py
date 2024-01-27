import pymysql
import traceback
import gensim
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models.word2vec import Text8Corpus
import numpy as np
from itertools import permutations
from pybliometrics.scopus import AuthorRetrieval
from pybliometrics.scopus import CitationOverview
import pandas as pd
import time
import math
#from transformers import  BertModel, BertConfig,BertTokenizer
#import torch
#import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
#from pytorch_pretrained import BertModel, BertTokenizer


#train model
'''
sentences = word2vec.Text8Corpus('D:/LSH/毕业/word2vec/dataset_all.txt')
model = word2vec.Word2Vec(sentences)
model.save('D:/LSH/毕业/word2vec/dataset_all.model')
sentences = word2vec.Text8Corpus('D:/LSH/毕业/word2vec/dataset12-14.txt')
model = word2vec.Word2Vec(sentences)
model.save('D:/LSH/毕业/word2vec/dataset12-14.model')
'''

#model1=Word2Vec.load('D:/LSH/毕业/word2vec/dataset.model')
#print((model1.wv['algorithm']))

#for e in model1.wv.most_similar(positive=['algorithm'],topn=10):
#    print(e[0],e[1])

#average semantic span
def Get_aver_distance():
    sql = "select ID,Abstract from dataset where `Year`='2011' or `Year`='2012'"
    results = DB_Connect(sql, "select")
    for result in results:
        str1 = result[1].strip()
        ls = str1.split(".")
        length = len(ls)
        if length < 4:
            aver_turn = 1
        else:
            distance = 0
            # current semantic span
            for i in range(length - 1):
                distance += Get_Distance(ls[i], ls[i + 1])
            distance_al = []  # semantic span between each spot
            for i in range(length):
                distance_al.append([])
                for j in range(length):
                    dis = Get_Distance(ls[i], ls[j])
                    distance_al[i].append(dis)
            index = []
            for i in range(1, length - 1):
                index.append(i)
            tsp = []  # semantic span of all possible span
            for index_per in permutation(index):
                distance_a = 0
                distance_a += distance_al[0][index_per[0]]
                for j in range(len(index_per) - 1):
                    distance_a += distance_al[index_per[j]][index_per[j + 1]]
                distance_a += distance_al[index_per[-1]][-1]
                tsp.append(distance_a)
            aver_span = round(distance / min(tsp), 2)

        # distance=np.around(distance/length,2)
        sql1 = "update dataset set Aver_turn = " + str(aver_span) + " where ID=" + str(result[0])
        DB_Connect(sql1, "update")

#average semantic inflection
def Get_aver_turn():
    sql="select ID,Abstract from dataset "
    results=DB_Connect(sql,"select")
    for result in results:
        ls=result[1].split('.')
        if '©' in result[1]:
            ls=ls[0:-1]
        ls=ls[0:-1]
        length=len(ls)
            if length < 4:
                aver_turn = 1
            else:
                distance = 0
            # current semantic span
                for i in range(length - 1):
                    distance += Get_Distance(ls[i], ls[i + 1])
                distance_al = []  # semantic span between each spot
                for i in range(length):
                    distance_al.append([])
                    for j in range(length):
                        dis = Get_Distance(ls[i], ls[j])
                        distance_al[i].append(dis)
                index = []
                for i in range(1, length - 1):
                    index.append(i)
                tsp = []  # semantic span of all possible span
                for index_per in permutations(index):
                    distance_a = 0
                    distance_a += distance_al[0][index_per[0]]
                    for j in range(len(index_per) - 1):
                        distance_a += distance_al[index_per[j]][index_per[j + 1]]
                    distance_a += distance_al[index_per[-1]][-1]
                    tsp.append(distance_a)
                aver_turn = round(distance / min(tsp), 2)
            
           
        sql1 = "update dataset set Aver_turn = " + str(aver_turn) + " where ID=" + str(result[0])
        DB_Connect(sql1, "update")

#average academic age and h-index
def Get_age_aver():
    sql="select ID,Year,Author_ID from dataset where aver_h_index is null"
    results =DB_Connect(sql,"select")
    for result in results:
        author=result[2].split(";")
        author.pop()
        age=0
        h_index=0
        count=0
        for aur in author:
            if count%100==0:
                time.sleep(10)
            count+=1
            try:
                au=AuthorRetrieval(aur,refresh=True)
                tu=au.publication_range
                age+=(int(result[1])-tu[0])
                h_index += au.h_index
            except Exception as ex:
                #print(sql)
                time.sleep(10)
                traceback.print_exc()
            age=round(age/len(author),2)
        h_index = round(h_index / len(author), 2)
        sql2="update dataset set acd_age_aver="+str(age)+", " \
             "aver_h_index="+str(h_index)+" where ID="+str(result[0])
        DB_Connect(sql2,"update")
#citation
def Get_citation():
    sql="select ID,EID,`Year` from dataset where fifth_cit is null"
    results=DB_Connect(sql,"select")
    try:
        for result in results:
            eid=[]
            eid.append(result[1].split('-')[2])
            if len(result[1].split('-'))!=3:
                print(result[0])
                continue
            year=int(result[2])
            co = CitationOverview(eid, start=year,end=year+5)
            ls=co.cc
            st=ls[0][0][1]+ls[0][1][1]
            sql2="update dataset set first_cit="+str(st)+",second_cit="+str(ls[0][2][1])+"," \
                 "third_cit="+str(ls[0][3][1])+",fourth_cit="+str(ls[0][4][1])+",fifth_cit="+str(ls[0][5][1])+\
                 " where ID="+str(result[0])
            DB_Connect(sql2,"update")
    except Exception as ex:
        time.sleep(10)
        print(result[0])
        traceback.print_exc()

#average academic papers
def Get_acdemic_aver():
    sql = "select ID,Year,Author_ID from dataset where ID>971"
    results = DB_Connect(sql, "select")
    for result in results:
        author = result[2].split(";")
        author.pop()
        paper = 0
        i=0
        for aur in author:
            i+=1
            if i%3==0:
                time.sleep(5)
            au = AuthorRetrieval(aur)
            docs = pd.DataFrame(au.get_documents())
            docs['year'] = docs['coverDate'].str[:4]
            se=docs['year'].value_counts().sort_index()
            for i in range(len(se)):
                if se.index[i] > result[1]:
                    break
                else:
                    paper+=se.values[i]
        paper = round(paper / len(author), 2)
        sql2 = "update dataset set acd_aver=" + str(paper) + " ," \
                " where ID=" + str(result[0])
        DB_Connect(sql2, "update")

#citescore
def Get_Journal_Impact():
    path="C:/Users/Dell/Desktop/journal.txt"
    with open(path,encoding='utf-8') as file:
        for result in file:
            paper=result.strip().split('\t')
            for i in range(2011,2016):
                if paper[i-2010]!='null':
                    sql="update dataset set Journal_impact="+str(paper[i-2010])+" where " \
                        "Source_publication_name='"+paper[0]+"' and `Year`="+str(i)
                DB_Connect(sql,"update")

#author_number
def Get_author_num():
    sql1="select ID,Author_ID from dataset where Author_num is null"
    results=DB_Connect(sql1,"select")
    for result in results:
        author=result[1].split(';')
        author_num=len(author)-1
        sql2="update dataset set Author_num="+str(author_num)+" where ID="+str(result[0])
        DB_Connect(sql2,"update")

#reference number
def Get_ref_num():
    sql = "select ID, reference from dataset where reference_num is null"
    results = DB_Connect(sql, "select")
    for result in results:
        str1 = result[1]
        if not str1:
            ref_num = 0
        else:
            ls = str1.split(";")
            ref_num = len(ls)
        sql2 = "update dataset set reference_num=" + str(ref_num) + " where ID=" + str(result[0])
        DB_Connect(sql2, "update")

def Get_Distance(str1,str2):
    size=100
    w2v=Word2Vec.load('D:/LSH/毕业/word2vec/dataset_all.model')
    vec1=bulid_sentence_vector(str1,size,w2v)
    vec2=bulid_sentence_vector(str2,size,w2v)
    distance=cosine_sim(vec1,vec2)
    return distance

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

def cosine_sim(vec1,vec2):
    a=np.array(vec1)
    b=np.array(vec2)
    cos1=np.sum(a*b)
    cos21=np.sqrt(sum(sum(a**2)))
    cos22 = np.sqrt(sum(sum(b ** 2)))
    if cos21*cos22 ==0:
        cosine_value=0
    else:
        cosine_value=(cos1/float(cos21*cos22))
    return cosine_value

def permutation(li):
    return itertools.permutations(li)

def DB_Connect(sql,flag):
    results = []
    try:
        if flag == "select":
            cursor.execute(sql)
            results = cursor.fetchall()
        if flag == "one":
            cursor.execute(sql)
            results = cursor.fetchone()
        if flag == "update":
            cursor.execute(sql)
            db.commit()
        return results
    except Exception as ex:
        print(sql)
        traceback.print_exc()

def dataprocess():
  
    sql="(select * from (select ID,row_number() over( order by citation_count) ran," \
        "Title,Abstract from dataset) t1 order by rand()"
    results =DB_Connect(sql,"select")
    for result in results:
        if int(result[1])%5==2:
            with open("/dev.txt", 'a', encoding='utf-8') as f:
                f.write(text)
                f.write("\n")
        elif int(result[1])%5==3:
            with open("/test.txt", 'a', encoding='utf-8') as f:
                f.write(text)
                f.write("\n")
        else:
            with open("/train.txt", 'a', encoding='utf-8') as f:
                f.write(text)
                f.write("\n")
            

if __name__=='__main__':
    db = pymysql.connect(host="localhost", user="root", password="123456", database="Dataset")
    cursor = db.cursor()
    Get_age_aver()
    Get_citation()
    Get_Journal_Impact()
    Get_ref_num()
    Get_author_num()
    Get_aver_distance()
    Get_aver_turn()
    Get_age_aver()
    Get_acdemic_aver()
    dataprocess()
   









