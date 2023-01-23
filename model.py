import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pickle
from datetime import datetime

seed = 42
iter = 5000

f = open("results.txt","a+")
now = datetime.now()
print(now, file= f)

def fit_lr(train_X, train_y,test_X,test_y):
  op_dict = {}
  print("========logistic regression============", file=f)
  print("========logistic regression============")
  logistic = LogisticRegression(random_state=seed, class_weight= 'balanced',max_iter=iter)
  logistic.fit(train_X, train_y)
  l_pred_y = logistic.predict(test_X)
  report = classification_report(test_y, l_pred_y)
  print(report, file = f)
  print(report)

  op_dict['pred_label'] = l_pred_y
  op_dict['classification_report'] =report
  op_dict['model'] = logistic
  
  return op_dict


def fit_svm(train_X, train_y,test_X,test_y):

  op_dict = {}
  print("========SVM============", file=f)
  print("========SVM============")
  svc = SVC(gamma='auto', kernel= 'linear',class_weight = 'balanced', random_state = seed, max_iter = iter)
  svc.fit(train_X, train_y)
  sv_pred_y = svc.predict(test_X)
  report = classification_report(test_y, sv_pred_y)
  print(report)
  print(report,file=f)

  op_dict['pred_label'] = sv_pred_y
  op_dict['classification_report'] =report
  op_dict['model'] = svc
  
  return op_dict

def fit_rf(train_X, train_y,test_X,test_y):
  op_dict = {}
  print("========Random Forest-tfidf============", file=f)
  print("========Random Forest-tfidf============")
  rf = RandomForestClassifier(class_weight = 'balanced', random_state = seed, criterion = 'entropy')
  rf.fit(train_X, train_y)
  rf_pred_y = rf.predict(test_X)
  report = classification_report(test_y, rf_pred_y)
  print(report, file = f)
  print(report)

  op_dict['pred_label'] = rf_pred_y
  op_dict['classification_report'] =report
  op_dict['model'] = rf
  return op_dict



def fit_model(embeddings, model_choice="SVM", emb_choice="tfidf", model = None, mode= "train_eval"):
  model_op = {}
  if mode == "train_eval":
    if emb_choice == "tfidf":
      train_X = embeddings['train']['tfidf']['matrix']
      test_X = embeddings['valid']['tfidf']['matrix']
      
      train_y = embeddings['train']['label_info']['train_sentence_labels']
      test_y = embeddings['valid']['label_info']['valid_sentence_labels']
      
      print("shape of tfidf", train_X.shape)
    
    if model_choice == "LR":
      model_op[model_choice]  = fit_lr(train_X,train_y,test_X,test_y)
      embeddings['valid']['model_op'] = model_op
      
    if model_choice == "SVM":
      model_op[model_choice] = fit_svm(train_X,train_y,test_X,test_y)
      embeddings['valid']['model_op'] = model_op
    
    if model_choice == "RF":
      model_op[model_choice]  = fit_rf(train_X,train_y,test_X,test_y)
      embeddings['valid']['model_op'] = model_op
    
    if model_choice == "all_discrete":
      
      model_op['LR'] = fit_lr(train_X,train_y,test_X,test_y)
      model_op['SVM'] = fit_svm(train_X,train_y,test_X,test_y)
      model_op['RF'] = fit_rf(train_X,train_y,test_X,test_y)
      embeddings['valid']['model_op'] = model_op
     
  if mode == "infer":
    model_op = {}
    test_X = embeddings['tfidf']['matrix']
    model_op['pred_label'] = model.predict(test_X)
    embeddings['model_op'] = model_op

  
  
  return embeddings
  
# def get_output_csv(embeddings):


#   gt = list(embeddings['valid']['label_info']['sentence_labels'])
#   op_dict = 
#   df1 = pd.DataFrame.from_dict()
#   df1['gt'] = gt
#   df1["id"] = embeddings['valid']['sent_ids']
#   df1['sentence'] = embeddings['valid']['sentences']
#   df1.to_csv("preds_val.csv")

