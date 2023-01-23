README

Artefacts
- The trained SVM model : SVMradrec_cls.model
- tfidf vectorizer: tfidf.vectorizer


Model Training
- pipeline.py trains a model to perform sentence classification

Model inference
- To use the provided trained model, please use infer.py
- infer.py 
  Returns a dictionary
    key: txt file name 
    value: list with pairs containing (start, end) character indices 
    of recommendation sentences that were predicted