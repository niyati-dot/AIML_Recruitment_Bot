from pymongo import MongoClient
from datetime import datetime,timedelta
from pyshorteners import Shortener 
import json
import requests
import pandas as pd
import gridfs
from sseclient import SSEClient
from transformers import pipeline

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize  # to split sentences into words
from nltk.corpus import stopwords  # to get a list of stopwords
from collections import Counter  # to get words-frequency
import requests  # this we will use to call API and get data
import nltk
import json
from keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import Parallel, delayed
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
tokenizer = Tokenizer(num_words=5000)


#Function to initiate database connection
def getDbConnection():
    try:
        
        print("getDbConnection")
        client = MongoClient('mongodb://localhost:27017/RecruitmentBot')
        print("[INFO] Database connection successful")
        return {'success':True, 'connection':client}
    except Exception as ex:
        print("[ERROR] Database connection failed",exc_info=True)
        return {'success':False}




def insertData(df_json):
    print("[info] in insertData")
    try:
        dbConfig =  getDbConnection()
        if dbConfig['success']==True:
            client = dbConfig['connection']
            mydb=client['RecruitmentBot']
            excelDataCollection=mydb['questions']
            print(df_json)
            newDict={"questions":df_json}
            x=excelDataCollection.insert_one(newDict)
            print(x)
            dbConfig['connection'].close()
            response = {'status': True, "msg":"Successfully inserted data"}
            
        else:
            print("[error] unable to connect to database")
            response = {'status': False, "msg":"We are unable to connect to our database. Please try after sometime"}

    except Exception as ex:
        print("[error] Error Occured in insertData")
        print(ex)
        response = {
            "status":False,
            "msg":"[error] occured while inserting data"
        }
    return response
    
def trainModel():
    print("[info] in trainModel")
    try:
        print("try")
        nlp = pipeline("sentiment-analysis")
        dbConfig =  getDbConnection()
        if dbConfig['success']==True:
            client = dbConfig['connection']
            mydb=client['pythonProject']
            excelDataCollection=mydb['excelData']
            trainRecords=excelDataCollection.find()
            trainList=[]
            for i in range(0,trainRecords.count()):
                trainList.append(trainRecords[i]['Review'])
            traindf = pd.DataFrame(trainList)
                
            if(trainRecords.count()==0):
                print("[INFO] No Data present .")
                response={'status': False, "msg":"No data available to train"}
            else:
                print("[INFO] trainModel success")
                trainRecords.close()
                print(traindf)
                sentiment = []
                for j in trainList:
                    #sentiment.append(nlp(j))
                    sentimentDict={"Review":j,"Sentiment":nlp(j)}
                    sentiment.append(sentimentDict)
                print(sentiment)
                for sentiments in sentiment:
                    print(sentiments['Sentiment'][0]['label'])
                    sentimentDataCollection=mydb['sentimentData']
                    sentimentdatacollectionDict={"review":sentiments['Review'],"sentiment":sentiments['Sentiment'][0]['label']}
                    x=sentimentDataCollection.insert_one(sentimentdatacollectionDict)
                print(x.inserted_id)
                response = {'status': True, "msg":"Successfully trained"}
            dbConfig['connection'].close()
            
        else:
            print("[error] unable to connect to database")
            response = {'status': False, "msg":"We are unable to connect to our database. Please try after sometime"}

    except Exception as ex:
        print("[error] Error Occured in trainModel")
        print(ex)
        response = {
            "status":False,
            "msg":"[error] occured while trainModel"
        }
    return response
    
    
   
    
    
def getQuestion(question):
    print("[info] in getQuestion")
    try:
        dbConfig =  getDbConnection()
        if dbConfig['success']==True:
            client = dbConfig['connection']
            mydb=client['RecruitmentBot']
            questionCollection=mydb['questions']
            cur=questionCollection.find({'questionid':question})
            questionList=[]
            for i in range(0,cur.count()):
                questionDict={'question':cur[i]['question']}
                questionList.append(questionDict)
            
            cur.close()
            print(questionList[0])
            dbConfig['connection'].close()
            response = {
                "status":True,
                "msg":"Successfully",
                "data":questionList[0]['question']
            }
            
        else:
            response = {'status': False, "msg":"We are unable to connect to our database. Please try after sometime"}

    except Exception as ex:
        print("[error] Error Occured in getQuestion")
        print(ex)
        response = {
            "status":False,
            "msg":"[error] occured while getQuestion"
        }
    return response
    
    
def compareResponse(inputText,qno):
    print("[info] in compareResponse")
    try:
        dbConfig =  getDbConnection()
        if dbConfig['success']==True:
            client = dbConfig['connection']
            mydb=client['RecruitmentBot']
            questionCollection=mydb['questions']
            cur=questionCollection.find({'questionid':qno})
            answerList=[]
            for i in range(0,cur.count()):
                answerDict={'answer':cur[i]['answer']}
                answerList.append(answerDict)
            
            cur.close()
            answer = answerList[0]['answer']
            
            #answer = json.dumps(answer)
            print(answer)
            # df = pd.read_excel('chatbot-dataset.xlsx')
            # for i in range(len(df)-1):
                # model = SentenceTransformer('bert-base-nli-mean-tokens')
                # sen_embeddings_question= model.encode(df.loc[i][2:].to_list())
                # joblib.dump(model, f'Question{df["ID"][i]}.pkl')
            model = SentenceTransformer('bert-base-nli-mean-tokens')
            sen_embeddings_question = joblib.load("sen_embedding.pkl")
            sentence_to_test = model.encode([inputText])
            similarity = cosine_similarity(
            [sentence_to_test[0]],
            sen_embeddings_question
            )
            
            
            
            print(np.mean(similarity))
            # if inputText.lower() == answer.lower():
                # profileCollection=mydb['userProfile']
                # newDict={"questions":qno,"evaluation":"correct"}
                # profileCollection.insert_one(newDict)
                # print("inserted")
            # else:
                # profileCollection=mydb['userProfile']
                # newDict={"questions":qno,"evaluation":"wrong"}
                # profileCollection.insert_one(newDict)
                # print(" not inserted")
            if np.mean(similarity)> 0.5 and np.mean(similarity) < 0.7:
                profileCollection=mydb['userProfile']
                newDict={"questions":qno,"evaluation":0.5}
                profileCollection.insert_one(newDict)
                print("inserted")
            elif np.mean(similarity)> 0.7:
                profileCollection=mydb['userProfile']
                newDict={"questions":qno,"evaluation":1}
                profileCollection.insert_one(newDict)
                print("inserted")
            else:
                profileCollection=mydb['userProfile']
                newDict={"questions":qno,"evaluation":0}
                profileCollection.insert_one(newDict)
                print(" not inserted")
            response = {'status': True, "msg":"Successful"}
            
        else:
            print("[error] unable to connect to database")
            response = {'status': False, "msg":"We are unable to connect to our database. Please try after sometime"}

    except Exception as ex:
        print("[error] Error Occured in compareResponse")
        print(ex)
        response = {
            "status":False,
            "msg":"[error] occured while inserting data"
        }
    return response
    
    
def insertSentiment(inputText):
    print("[info] in insertSentiment")
    try:
        dbConfig =  getDbConnection()
        if dbConfig['success']==True:
            client = dbConfig['connection']
            mydb=client['RecruitmentBot']
            sentimentCollection=mydb['userSentiment']
            tw = tokenizer.texts_to_sequences([inputText])
            tw = pad_sequences(tw,maxlen=200)
            model = load_model("senti_model.h5")
            prediction = int(model.predict(tw).round().item())
            
            if prediction ==0:
                newDict={"feedback":inputText,"sentiment": "Negative"}
                sentimentCollection.insert_one(newDict)
                print("inserted")
                response = {'status': True, "msg":"Successful"}
            else:
                newDict={"feedback":inputText,"sentiment":"Positive"}
                sentimentCollection.insert_one(newDict)
                print("inserted")
                response = {'status': True, "msg":"Successful"}
        else:
            print("[error] unable to connect to database")
            response = {'status': False, "msg":"We are unable to connect to our database. Please try after sometime"}

    except Exception as ex:
        print("[error] Error Occured in insertSentiment")
        print(ex)
        response = {
            "status":False,
            "msg":"[error] occured while inserting data"
        }
    return response
    
    

    


    
    
  