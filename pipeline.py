#default libraries
import os
import glob
from pathlib import Path
import logging
import tqdm
from collections import OrderedDict, Counter
import spacy
import numpy as np
import pickle
import pandas as pd

#programs in folder
import convert_to_spert as SP
from embedding import embedding_fn
from transformers import AutoModel, AutoTokenizer
from model import fit_model




if __name__ == "__main__":

  #read data
  #source_path = r"D:\\projects\\radiology\\doc_classification\\data\\"
  #val_path = r"D:\\projects\\radiology\\doc_classification\\data\\"

  source_path = r"/home/gramacha/radiology/recommendations/analysis/doc_classification/data/train/"
  val_path = r"/home/gramacha/radiology/recommendations/analysis/doc_classification/data/validation/"
  
  annotations = SP.import_brat_dir(source_path)
  val_annotations = SP.import_brat_dir(val_path)

  doc_names = [x[0] for x in annotations]
  doc_level = {dname:[] for dname in doc_names}

  val_doc_names = [x[0] for x in val_annotations]
  val_doc_level = {dname:[] for dname in val_doc_names}



  spacy_model='en_core_web_sm'
  tokenizer = spacy.load(spacy_model)

  allowable_tb = SP.get_allowable_types(None)

  logging.info(f"")
  logging.info(f"Converting documents: {source_path}")
  pbar = tqdm.tqdm(total=len(annotations))

  # Loop on annotated files
  converted_docs = []
  val_converted_docs = []


  for id, text, ann in annotations:
      doc = SP.convert_doc(text, ann, id, tokenizer, allowable_tb=allowable_tb)
      converted_docs.extend(doc)
      # formatted_docs.append(format_doc(doc))

      pbar.update(1)
  pbar.close()

  for id, text, ann in val_annotations:

      doc = SP.convert_doc(text, ann, id, tokenizer, allowable_tb=allowable_tb)
      val_converted_docs.extend(doc)
      # formatted_docs.append(format_doc(doc))

      pbar.update(1)
  pbar.close()

  print(len(converted_docs))
  print(len(val_converted_docs))

  doc_sents =[]
  label_info = {}
  label_info['label_dict'] = {1:'REC', 0: 'NOT_REC'}
  label_info['other_rec_labels'] = ["IMPT_REC","IMPT_REC_MISS_UNLIKELY","IMPT_CONTINGENT_REC","UNIMPORTANT_REC"]

  b_tokenizer = AutoTokenizer.from_pretrained("/home/gramacha/radiology/recommendations/analysis/doc_classification/model/Bio_ClinicalBERT/")
  embedding_model = AutoModel.from_pretrained("/home/gramacha/radiology/recommendations/analysis/doc_classification/model/Bio_ClinicalBERT")
  bert_embed = {'tokenizer':b_tokenizer, 'embedding_model':embedding_model}
  modified_docs = []
  
  doc_level_dict = {'train':doc_level, 'valid':val_doc_level}
  converted_docs_dict = {'train':converted_docs, 'valid':val_converted_docs}

  embeddings= embedding_fn(doc_level_dict, converted_docs_dict,label_info,bert_embed,"tfidf",mode = "train_eval")
  
  
  '''
  pickle input files
  '''
  with open("embeddings.pickle", "wb") as f:
    pickle.dump(embeddings, f)
  
  
  embeddings_with_pred = fit_model(embeddings, model_choice = "SVM", emb_choice= "tfidf", mode="train_eval")
  
  
  with open("embeddings_with_pred.pickle", "wb") as f:
    pickle.dump(embeddings_with_pred, f)
  

 



