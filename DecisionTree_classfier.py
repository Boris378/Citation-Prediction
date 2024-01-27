from sklearn.tree import DecisionTreeClassifier
import pymysql
import gensim
from gensim.models import Word2Vec
import numpy as np
import traceback

def DB_Connect(sql,flag):
    results = []
    try:
        if flag == "select":
            cursor.execute(sql)
            results = cursor.fetchall()
        if flag == "update":
            cursor.execute(sql)
            db.commit()
        return results
    except Exception as ex:
        print(sql)
        traceback.print_exc()

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
def two_classfier_all():
    path = "/dataset_all.txt"
    with open(path, encoding='utf-8') as file:
        for paper in file:
            paper_info = paper.split("	")
            vec = np.zeros(size).reshape(1, size)
            sql1 = "select Aver_span,Aver_turn ,aver_Author_num,reference_num,acd_age_aver,aver_h_index,Journal_Impact,papers,type,Title," \
                   "Abstract from dataset where ID=" + str(paper_info[0])
            results = DB_Connect(sql1, "select")
            for result in results:
                str1 = result[6] + '.' + result[7]
                str1_ls = str1.split('.')
                for str2 in str1_ls:
                    vec += bulid_sentence_vector(str2, size, w2v)
                temp = []
                for i in range(100):
                    temp.append(vec[0][i])

                for i in range(6):
                   temp.append(result[i])
                #temp.append(result[0])
                if int(paper_info[1]) <= 3532:  # HCP
                    if int(paper_info[0]) % 5 == 2:  # test
                        test.append(temp)
                        test_re.append(1)
                    else:  # train 
                        X.append(temp)
                        y.append(1)
                else:  # non-HCP
                    if int(paper_info[0]) % 5 == 2:
                        test.append(temp)
                        test_re.append(0)
                    else:
                        X.append(temp)
                        y.append(0)

    dt_model = DecisionTreeClassifier(class_weight='balanced',min_samples_leaf=5,splitter='best',max_depth=4)  
    dt_model.fit(X, y)  
    TP, FN, FP, TN = 0, 0, 0, 0
    count = 0
    for i in range(len(test_re)):
        temp = []
        temp.append(test[i])
        re = dt_model.predict(temp)
        if re != 0 and re != 1:
            count += 1
        if test_re[i] == 1 and re == 1:
            TP += 1
        if test_re[i] == 1 and re == 0:
            FN += 1
        if test_re[i] == 0 and re == 1:
            FP = +1
        if test_re[i] == 0 and re == 0:
            TN += 1
    print(count)
    precise = round(TP / (TP + FP), 4)
    recall = round(TP / (TP + FN), 4)
    print("F1-score：" + str(round(2 * precise * recall / (precise + recall), 4)))
    print("accuracy：" + str(round((TP + TN) / (TP + FP + TN + FN), 4)))

if __name__=='__main__':
    X, y, test, test_re = [], [], [], []  # train,train_flag,test,test_flag
    size = 100
    w2v = Word2Vec.load('/dataset_all.model')
    db = pymysql.connect(host="localhost", user="root", password="123456", database="Dataset")
    cursor = db.cursor()
    two_classfier_all()
