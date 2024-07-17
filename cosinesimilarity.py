import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel('chatbot-dataset.xlsx')

for i in range(len(df)-1):
  model = SentenceTransformer('bert-base-nli-mean-tokens')
  sen_embeddings_question= model.encode(df.loc[i][2:].to_list())
  joblib.dump(model, f'Question{df["ID"][i]}.pkl')

joblib.dump(sen_embeddings_question, f'sen_embedding.pkl')

print("model created")