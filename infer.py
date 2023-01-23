import pickle
import convert_to_spert as SP
from embedding import embedding_fn
import argparse
import spacy
import tqdm
from model import fit_model

def infer_model(model, embedding):
  pass

if __name__ == "__main__":

  # arg_parser = argparse.ArgumentParser()
  # arg_parser.add_argument('--source_path', type=str, help="Path to txt and ann files")
  # arg_parser.add_argument('--dest_path', type=str, help="Destination to store predictions ")
  # arg_parser.add_argument('--model_path', type=str, help="cls_model path")

  # args = arg_parser.parse_args()

  rec_labels = ["IMPT_REC","IMPT_REC_MISS_UNLIKELY","IMPT_CONTINGENT_REC","UNIMPORTANT_REC"]

  # with open(args.model_path, "rb") as g:
  #   embedding_with_pred = pickle.load(g)
  
  
  with open("embeddings_with_pred.pickle", "rb") as g:
    embedding_with_pred = pickle.load(g)


  tfidfvectorizer = embedding_with_pred['train']['tfidf']['vectorizer']
  model_op = embedding_with_pred['valid']['model_op']
  cls_model = model_op['SVM']['model']



  # annotations = SP.import_brat_dir(args.source_path)
  # pbar = tqdm.tqdm(total=len(annotations))

  #modify source path containing text and ann files. need to provide empty ann files for parsing
  source_path = r"/home/gramacha/radiology/recommendations/analysis/doc_classification/data/validation/"

  annotations = SP.import_brat_dir(source_path)
  pbar = tqdm.tqdm(total=len(annotations))

  spacy_model='en_core_web_sm'
  tokenizer = spacy.load(spacy_model)
  allowable_tb = SP.get_allowable_types(None)


  converted_docs = []
  for id, text, ann in annotations:
    doc = SP.convert_doc(text, ann, id, tokenizer, allowable_tb=allowable_tb)
    converted_docs.extend(doc)
    # formatted_docs.append(format_doc(doc))

    pbar.update(1)
  pbar.close()

  doc_names = [x[0] for x in annotations]
  doc_level = {dname:[] for dname in doc_names}
  label_infor = {'other_rec_labels': rec_labels}
  
  embeddings = embedding_fn(doc_level, converted_docs, label_info = label_infor, mode = "infer",vectorizer= tfidfvectorizer )
  op_embeddings_with_pred = fit_model(embeddings, model= cls_model,mode="infer")
  
  # with open(args.dest_path+"//embeddings_with_pred.pickle", "wb") as g:
  #   pickle.dump(embeddings_with_pred_new.pickle,g)
  

  with open("embeddings_with_pred_new.pickle", "wb") as g:
    pickle.dump(op_embeddings_with_pred,g)

  char_indices = {}

  for i,_ in enumerate(op_embeddings_with_pred['sentences']):
    doc_id = op_embeddings_with_pred['sent_ids'][i].split('[')[0]
    char_indices[doc_id] = []

    if op_embeddings_with_pred['model_op']['pred_label'][i] == 1:
      char_start =converted_docs[i]['offsets'][0][0]
      char_end =converted_docs[i]['offsets'][-1][1]
      pair = (char_start,char_end)
      char_indices[doc_id].append(pair)



  #     for e in converted_docs[i]['entities']:
  #       if e['type'] in rec_labels:
  #         t_start = e['start']
  #         t_end = e['end']-1
  #         char_start = converted_docs[i]['offsets'][t_start][0]
  #         char_end = converted_docs[i]['offsets'][t_end][1]

  print(char_indices)

  #returns a dictionary with key as the txt file name and value as a list with pairs containing start 
  # and end character indices of recommendation sentences that were predicted
  with open("rec_sent_indices.pickle", "wb") as g:
    pickle.dump(char_indices,g)


  
  

