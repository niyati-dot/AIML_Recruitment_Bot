import os
import flask  
import pickle
from flask import Flask, render_template, request,send_from_directory
from flask import Flask, session, redirect, url_for, escape, request
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import datetime
from datetime import time, date
import dateutil.parser
import requests
from helper import dbHandler
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import json


UPLOAD_FOLDER=r'\Lambton\Sem 3\AML 3206\Project\documents'
app=Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}}) 
app.secret_key='f1389197c451294efa40302c75cdde1d'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

lemmatizer = WordNetLemmatizer()

# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("data.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
# intentIdentified=False
# state="init"
# intent={}
@app.route('/')
def home():
    print("clearing session")
    session.clear()
    session['intentIdentified']=False
    session['state']="init"
    session['intent']='none'
    now = datetime.datetime.now()
    now = now.strftime("%m/%d/%Y, %H:%M:%S")
    t=datetime.datetime.now()
    d = dateutil.parser.parse(now).date()
    print("new session :",session)
    
    return render_template('ChatBotUI.html',d=d,t=t)
    #return render_template('index.html',d=d,t=t)
	


# @app.route('/train',methods = ['GET','POST'])
# def train():
    # #if request.method == 'POST':
     # #UPLOAD_FOLDER=r'\Users\606259\Desktop\SaleAssist'
     # print("train")
     # # UPLOAD_FOLDER=r'\chat-bot_assessment\WebServer\src'
     # # data = request.files['file'] 
     # # filename=data.filename 
     # # data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

     # # Importing the dataset
     # dataset = pd.read_csv('data.csv')
     # #X = dataset.iloc[:,0].values
     # #y = dataset.iloc[:,1].values
     # X=dataset['Input Text']
     # y=dataset['Intent']
     # print(X)
     # print(y)

     



     # # Encoding the Dependent Variable
     # # labelencoder_x = LabelEncoder()
     # # X = labelencoder_x.fit_transform(X)
     # # labelencoder_y = LabelEncoder()
     # # y = labelencoder_x.fit_transform(y)
     # # X=X.reshape(-1,1)
     # # y=y.reshape(-1,1)
     # count_vect = CountVectorizer()
     # tfidf_transformer = TfidfTransformer()


     # skf = StratifiedKFold()
     # for train_index, test_index in skf.split(X, y):
      # X_train, X_test = X[train_index], X[test_index]
      # y_train, y_test = y[train_index], y[test_index]
      # print("----")
      # print(X_train)
      # print(type(X_train))
      # print(X_train.shape)
      # X_train_counts = count_vect.fit_transform(X_train)
      # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
      # # X_test_counts = count_vect.fit_transform(X_test)
      # # X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
      # # print(X_test_tfidf)
      # print(y_test)
      # # Fitting Decision Tree Classification to the Training set
      # # classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 100)
      # # classifier.fit(X_train_tfidf, y_train)
      # # score=classifier.score(X_train_tfidf,y_train)
      # # print("score",score)
       # # Fitting Random Forest Classification to the Training set
      # random = RandomForestClassifier(criterion = 'entropy', random_state = 100)
      # random.fit(X_train_tfidf, y_train)
      # rscore=random.score(X_train_tfidf,y_train)
      # print("score",rscore)
      # inputText=['how are you']
      # print(type(inputText))
      # inputText=count_vect.transform(inputText)
      # print("&&&")
      # inputText_tfidf = tfidf_transformer.transform(inputText)
      # print(random.classes_)
      # #print(random.classes_,random.predict_proba(inputText_tfidf))
      # predict=random.predict_proba(inputText_tfidf)

      # print(predict)
      # # X_test=count_vect.transform(X_test)
      # # X_test_tfidf = tfidf_transformer.transform(X_test)
      # with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\classifier.pickle', 'wb') as f:
        # pickle.dump(random, f)
      # with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\countvector.pkl', 'wb') as f:
        # pickle.dump(count_vect, f)
      # with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\tfidf.pkl', 'wb') as f:
        # pickle.dump(tfidf_transformer, f)

     # return ("training completed succesfully")

    # #else:
      # #return "invalid method"

# @app.route('/predict',methods = ['POST'])
# def predict():
	# print("session .... : ",session)
	# # Predicting the Test set results
	# with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\classifier.pickle', 'rb') as f:
		# random= pickle.load(f)
	# with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\countvector.pkl', 'rb') as f:
		# count_vect= pickle.load(f)
	# with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\tfidf.pkl', 'rb') as f:
		# tfidf_transformer= pickle.load(f)
		
	# print("model loaded ",random.classes_)
	# inputText=request.json["message"]
	
	# text=inputText
	# inputText=[inputText]
	# print("text",inputText)
	# #inputText=np.asarray(inputText)
	# dataset = pd.read_csv('data.csv')
	# X=dataset['Input Text']
	# y=dataset['Intent']
	# skf = StratifiedKFold()
	# for train_index, test_index in skf.split(X, y):
		# X_train, X_test = X[train_index], X[test_index]
		# y_train, y_test = y[train_index], y[test_index]

		# X_train_counts = count_vect.fit_transform(X_train)
		# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	# #
	# if(	session['intentIdentified'] == False):

		# inputText=count_vect.transform(inputText)
		# inputText_tfidf = tfidf_transformer.transform(inputText)
		# print(random.classes_)
		# #print(random.classes_,random.predict_proba(inputText_tfidf))
		# predict=random.predict_proba(inputText_tfidf)
		# session['intentIdentified']=True
		# print("Predict : ",predict[0][2])
		# if predict[0][0]>=0.7:
			# print("Predict : ",predict[0][0])
			# session['intent']="CoolerProblem"
		# elif predict[0][1]>=0.7:
			# print("Predict : ",predict[0][1])
			# session['intent']="Greeting"
		# elif predict[0][2]>=0.01:
			# print("Predict : ",predict[0][2])
			# session['intent']="GreetingResponseYes"
		# elif predict[0][3]>=0.9:
			# print("Predict : ",predict[0][3])
			# session['intent']="GreetingResponseNo"
		# else:
			# session['intent']="others"
	# # session['intent']=intent
  # #else:
  # #  intent=session['intent']
	# print(session)
	# if session['intent']=="Greeting":
		# #session['intentIdentified']=False
		# #session['state']="greeting"
		# if session['state']=="init" :
			# session['state']="greeting"
			# return "Hi, Hope you are doing well?"
		# elif session['state']=="greeting" :
			# if "good" in text.lower() or "alright" in text.lower() or "ok" in text.lower() or "yes" in text.lower():
				# session['intentIdentified']=False
				# session['state']="init"
				# return "That's good to hear. Hope you are ready to take the assessment?"
			# elif "no" in text.lower() or "not good" in text.lower():
				# session['intentIdentified']=False
				# session['state']="init"
				# return "Hope everything is ok?Are you comfortable taking the assessment today?"
	# elif session['intent']=="GreetingResponseNo":
			# session['intentIdentified']=False
			# session['state']="init"
			# #response = dbHandler.insertData(text)
			# return "Don't worry we will reschedule the assessment and mail you the updated assessment link."
	# elif session['intent']=="GreetingResponseYes":
			# #session['intentIdentified']=False
			# if session['state'] == "init":
				# #response = dbHandler.insertData(text)
				# session['state']="question 1"
				# return "That's Great!!! For your assessment you will be given a set of questions for which you should type in your response for.If you are ready to start the assessment type 'YES'."
			# elif session['state'] == "question 1":
				# if "yes" in text.lower():
					# session['state']="question 2"
					# question = 1
					# response = dbHandler.getQuestion(question)
					# print(response['data'])
					# question1 = response['data']
					# return question1
				# else:
					# return "If you are not ready we can reschedule the test"
			# elif session['state'] == "question 2":
					# session['state']="question 3"
					# question = 2
					# response = dbHandler.getQuestion(question)
					# print(response['data'])
					# question2 = response['data']
					# return question2
			# elif session['state'] == "question 3":
					# session['state']="question 4"
					# question = 3
					# response = dbHandler.getQuestion(question)
					# print(response['data'])
					# question3 = response['data']
					# return question3
			# elif session['state'] == "question 4":
					# session['state']="question 5"
					# question = 4
					# response = dbHandler.getQuestion(question)
					# print(response['data'])
					# question4 = response['data']
					# return question4
			# elif session['state'] == "question 5":
					# session['state']="end of question"
					# question = 5
					# response = dbHandler.getQuestion(question)
					# print(response['data'])
					# question5 = response['data']
					# return question5

	# elif session['intent']=="CoolerProblem":
		# if session['state']=="init" :
			# session['state']="problem check"
			# return "Ok, what problem you are facing?"
		# elif session['state']=="problem check":
			# session['problem']=text
			# print(session['problem'])
			# session['state']="ticket confirmation"
			# return ("Do you want to raise a ticket (yes/no) ?")
		# elif session['state'] =="ticket confirmation":
			# if "yes" in text.lower() or "ok" in text.lower():
				# #code to call api--> pass (session['problem']  and get ticket number
				# ticketnumber=12345
				# print("enter")
				# url='https://cognizantedusbpov.service-now.com/api/cogz/create_incident/incident_creator'
				# payload = {
				# "userid":"1234",
				# "tickettype":"incident",
				# "description":session['problem'],
				# "priority":"3",
				# "assignmentgroup":"SH LEVEL2 SUPPORT"}

				# headers = {'Content-type': 'application/json'}

				# r = requests.post(url, json=payload, headers=headers)
				# print(r)
				# print(payload)
				# val=r.json()
                
				# if val['result']['success']:
					# print(val)
					# ticket_id = val['result']['ticketid']
				# else:
					# ticket_id = None

				# session['state']='init'
				# session['intentIdentified']=False
				# flag=False
				# return "ticket raised succesfully . your ticket number is "+str(ticket_id)
			# elif "no" in text.lower():
				# session['state']='init'
				# session['intentIdentified']=False
				# return "thanks"
			# else:
				# return("respond with yes or no")
	# elif session['intent']=="others":
		# session['intentIdentified']=False
		# return "I am not able to understand"
        
 

# chat functionalities
def clean_up_sentence(sentence):
	print(sentence)
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
	return sentence_words
	
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
	# tokenize the pattern
	sentence_words = clean_up_sentence(sentence)
	# bag of words - matrix of N words, vocabulary matrix
	bag = [0] * len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				# assign 1 if current word is in the vocabulary position
				bag[i] = 1
				if show_details:
					print("found in bag: %s" % w)
	return np.array(bag)






 
@app.route('/train',methods = ['GET','POST'])
def train():
    #if request.method == 'POST':
     #UPLOAD_FOLDER=r'\Users\606259\Desktop\SaleAssist'
     print("train")
     # UPLOAD_FOLDER=r'\chat-bot_assessment\WebServer\src'
     # data = request.files['file'] 
     # filename=data.filename 
     # data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

     # Importing the dataset
     dataset = pd.read_csv('data.csv')
     #X = dataset.iloc[:,0].values
     #y = dataset.iloc[:,1].values
     X=dataset['Input Text']
     y=dataset['Intent']
     print(X)
     print(y)

     



     # Encoding the Dependent Variable
     # labelencoder_x = LabelEncoder()
     # X = labelencoder_x.fit_transform(X)
     # labelencoder_y = LabelEncoder()
     # y = labelencoder_x.fit_transform(y)
     # X=X.reshape(-1,1)
     # y=y.reshape(-1,1)
     count_vect = CountVectorizer()
     tfidf_transformer = TfidfTransformer()


     skf = StratifiedKFold()
     for train_index, test_index in skf.split(X, y):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      print("----")
      print(X_train)
      print(type(X_train))
      print(X_train.shape)
      X_train_counts = count_vect.fit_transform(X_train)
      X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
      # X_test_counts = count_vect.fit_transform(X_test)
      # X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
      # print(X_test_tfidf)
      print(y_test)
      # Fitting Decision Tree Classification to the Training set
      # classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 100)
      # classifier.fit(X_train_tfidf, y_train)
      # score=classifier.score(X_train_tfidf,y_train)
      # print("score",score)
       # Fitting Random Forest Classification to the Training set
      random = RandomForestClassifier(criterion = 'entropy', random_state = 100)
      random.fit(X_train_tfidf, y_train)
      rscore=random.score(X_train_tfidf,y_train)
      print("score",rscore)
      inputText=['how are you']
      print(type(inputText))
      inputText=count_vect.transform(inputText)
      print("&&&")
      inputText_tfidf = tfidf_transformer.transform(inputText)
      print(random.classes_)
      #print(random.classes_,random.predict_proba(inputText_tfidf))
      predict=random.predict_proba(inputText_tfidf)

      print(predict)
      # X_test=count_vect.transform(X_test)
      # X_test_tfidf = tfidf_transformer.transform(X_test)
      with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\classifier.pickle', 'wb') as f:
        pickle.dump(random, f)
      with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\countvector.pkl', 'wb') as f:
        pickle.dump(count_vect, f)
      with open(r'D:\Lambton\Sem 3\AML 3206\Project\documents\tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_transformer, f)

     return ("training completed succesfully")

    #else:
      #return "invalid method"

@app.route('/predict',methods = ['POST'])
def predict():
	print("session .... : ",session)
	# Predicting the Test set results
	inputText=request.json["message"]
	
	text=inputText
	print("text",inputText)
	if (session['state'] == "end of question") or (session['state'] == "question 2") or (session['state'] == "question 3") or (session['state'] == "question 4") or (session['state'] == "question 5"):
		if session['state'] == "question 2":
			qno = 1
		elif session['state'] == "question 3":
			qno = 2
		elif session['state'] == "question 4":
			qno = 3
		elif session['state'] == "question 5":
			qno = 4
		elif session['state'] == "end of question":
			qno = 5
		
		dbHandler.compareResponse(inputText,qno)

	if session['intentIdentified'] == False:
		
		p = bow(inputText, words, show_details=False)
		res = model.predict(np.array([p]))[0]
		ERROR_THRESHOLD = 0.40
		results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
		# sort by strength of probability
		results.sort(key=lambda x: x[1], reverse=True)
		return_list = []
		for r in results:
				return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
		
		print(return_list)
		#return return_list
		print(classes[r[0]])
		session['intent']= classes[r[0]]
		print(session['intent'])
		session['intentIdentified']=True
		print(session['intentIdentified'])
		# print("Predict : ",predict[0][2])
		# if predict[0][0]>=0.7:
			# print("Predict : ",predict[0][0])
			# session['intent']="CoolerProblem"
		# elif predict[0][1]>=0.7:
			# print("Predict : ",predict[0][1])
			# session['intent']="Greeting"
		# elif predict[0][2]>=0.01:
			# print("Predict : ",predict[0][2])
			# session['intent']="GreetingResponseYes"
		# elif predict[0][3]>=0.9:
			# print("Predict : ",predict[0][3])
			# session['intent']="GreetingResponseNo"
		# else:
			# session['intent']="others"
	# session['intent']=intent
	#else:
  #  intent=session['intent']
	print(session)
	
	if session['intent']=="Greeting":
		#session['intentIdentified']=False
		#session['state']="greeting"
		if session['state']=="init" :
			session['state']="greeting"
			return "Hi, Hope you are doing well?"
		elif session['state']=="greeting" :
			if "good" in text.lower() or "alright" in text.lower() or "ok" in text.lower() or "yes" in text.lower():
				session['intentIdentified']=False
				session['state']="init"
				return "That's good to hear. Hope you are ready to take the assessment?"
			elif "no" in text.lower() or "not good" in text.lower():
				session['intentIdentified']=False
				session['state']="init"
				return "Hope everything is ok?Are you comfortable taking the assessment today?"
	elif session['intent']=="GreetingResponseNo":
			session['intentIdentified']=False
			session['state']="init"
			#response = dbHandler.insertData(text)
			return "Don't worry we will reschedule the assessment and mail you the updated assessment link."
	elif session['intent']=="Help":
			session['intentIdentified']=False
			session['state']="init"
			#response = dbHandler.insertData(text)
			return "Please mail your queries to help@recruitmentBot.com.If you are facing technical issues while doing the assessment please call (xxx)-1234-453."
	elif session['intent']=="GreetingResponseYes":
			#session['intentIdentified']=False
			if session['state'] == "init":
				#response = dbHandler.insertData(text)
				session['state']="question 1"
				return "That's Great!!! For your assessment you will be given a set of questions for which you should type in your response for.If you are ready to start the assessment type 'YES'."
			elif session['state'] == "question 1":
				if "yes" in text.lower():
					session['state']="question 2"
					question = 1
					response = dbHandler.getQuestion(question)
					print(response['data'])
					question1 = response['data']
					return question1
				else:
					return "If you are not ready we can reschedule the test"
			elif session['state'] == "question 2":
					session['state']="question 3"
					question = 2
					response = dbHandler.getQuestion(question)
					print(response['data'])
					question2 = response['data']
					return question2
			elif session['state'] == "question 3":
					session['state']="question 4"
					question = 3
					response = dbHandler.getQuestion(question)
					print(response['data'])
					question3 = response['data']
					return question3
			elif session['state'] == "question 4":
					session['state']="question 5"
					question = 4
					response = dbHandler.getQuestion(question)
					print(response['data'])
					question4 = response['data']
					return question4
			elif session['state'] == "question 5":
					session['state']="end of question"
					question = 5
					response = dbHandler.getQuestion(question)
					print(response['data'])
					question5 = response['data']
					return question5
			elif session['state'] == "end of question":
					session['intentIdentified']=False
					session['state']="init"
					return "You have completed the assessment.Would you like to give a feedback regarding your experience using the RecruitmentBot?"

	elif session['intent']=="Review":
		if session['state']=="init" :
			session['state']="sentiment"
			return "Please type in your Feedback!!"
		elif session['state']=="sentiment":
			session['state']="end"
			dbHandler.insertSentiment(inputText)
			return "Thank you for your valuable Feedback!!"
		elif session['state']=="end":
			session['state']="end"
			return "You have completed the assessment,if you have any queries pleas mail us at help@recruitmentBot.com"
	elif session['intent']=="others":
		session['intentIdentified']=False
		return "I am not able to understand"
        

        


if __name__ == "__main__":
    app.run(debug=True)