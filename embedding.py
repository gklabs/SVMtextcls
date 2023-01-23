  
import torch
import numpy as np
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict, Counter

def extract_sentences(doc_level,converted_docs,label_info,dtype):
  modified_docs = []
  all_sent_ids = []
  all_sents_list = []
  doc_info ={}
  label_info[dtype+'_'+'sentence_labels'] = [0]* len(converted_docs)
  for id,sentence in enumerate(tqdm.tqdm(converted_docs)):
    for ent in sentence["entities"]:
      if ent['type'] in label_info['other_rec_labels']:
        label_info[dtype+'_'+'sentence_labels'][id] = 1
        break
    
    sent = ' '.join(sentence['tokens'])
    sentence['sent_text'] = sent
    all_sents_list.append(sent)
      
    sent_doc_id = sentence['id'].split('[')
    doc_id = sent_doc_id[0]
    sent_id = sent_doc_id[1]
    all_sent_ids.append(sentence['id'])
    doc_level[doc_id].append(sentence)
  
  
  doc_info['doc_level'] = doc_level

    
  cnt = Counter()

  # print(label_info)

  assert len(converted_docs) == len(label_info[dtype+'_'+'sentence_labels'])
  return all_sent_ids, all_sents_list,label_info, doc_info

def bertify(embeddings,bert_embed):
  X = np.empty(shape=[0,768])
  
  embeddings['bert']= {\
      "input_ids": torch.tensor(bert_embed['tokenizer'].encode(embeddings['sentences'],\
          padding=True,truncation=True, max_length=128,add_special_tokens = False)).unsqueeze(0)\
          }

  with torch.no_grad():
    vectors = bert_embed['embedding_model'](embeddings['bert']['input_ids'], output_attentions=True)['pooler_output'].detach().numpy()
    X = np.append(X,vectors,axis=0)
  
  return X,vectors

def embedding_fn(doc_level_dict,converted_docs_dict, label_info,bert_embed={},  e_type="tfidf", mode = "train_eval",vectorizer = None):

  if mode =="train_eval":
    train_embeddings = {}
    val_embeddings = {}

    train_embeddings['sent_ids'],train_embeddings['sentences'],train_embeddings['label_info'],train_embeddings['doc_info'] \
      = extract_sentences(doc_level_dict['train'],converted_docs_dict['train'],label_info,'train')

    
    val_embeddings['sent_ids'],val_embeddings['sentences'],val_embeddings['label_info'], val_embeddings['doc_info']=\
        extract_sentences(doc_level_dict['valid'],converted_docs_dict['valid'],label_info, 'valid')


    if e_type== "tfidf":
      vectorizer = TfidfVectorizer()
      tfidf = vectorizer.fit_transform(train_embeddings['sentences'])
      train_embeddings['tfidf']= {'matrix': tfidf, 'vectorizer': vectorizer}
      
      val_tfidfvector = vectorizer.transform(val_embeddings['sentences'])
      val_embeddings['tfidf'] = {'matrix': val_tfidfvector, 'vectorizer': vectorizer}
      

    if e_type == "bert":

      train_embeddings['bert'] ['matrix'] ,train_embeddings['bert']['vectors'] = bertify(train_embeddings,bert_embed)
      val_embeddings['bert'] ['matrix'] ,val_embeddings['bert']['vectors'] = bertify(val_embeddings,bert_embed)

    
    
    
    embeddings = {'train':train_embeddings, 'valid':val_embeddings}
  
  if mode == "infer":
    embeddings = {}
    embeddings['sent_ids'],embeddings['sentences'],embeddings['label_info'],embeddings['doc_info'] \
      = extract_sentences(doc_level_dict,converted_docs_dict,label_info,'test')
    
    if e_type== "tfidf":
      
      if vectorizer is not None:
        tfidf = vectorizer.transform(embeddings['sentences'])
        embeddings['tfidf']= {'matrix': tfidf, 'vectorizer': vectorizer}
      
      
      

  return embeddings





