import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import multiprocessing
# from bs4 import BeautifulSoup as bs
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
    
df = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
    
    

stopwords_list = stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()



def clean_data(text):
    text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
    text = re.sub(r'[\\/×\^\]\[÷]', '', text)
    return text

def change_lower(text):
    text = text.lower()
    return text

def remover(text):
    text_tokens = text.split(" ")
    final_list = [word for word in text_tokens if not word in stopwords_list]
    text = ' '.join(final_list)
    return text

def get_w2vdf(df):
    w2v_df = pd.DataFrame(df["review"]).values.tolist()
    for i in range(len(w2v_df)):
        w2v_df[i] = w2v_df[i][0].split(" ")
    return w2v_df

def train_w2v(w2v_df):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=4,
                         window=4,
                         vector_size=300, 
                         alpha=0.03, 
                         min_alpha=0.0007, 
                         sg = 1,
                         workers=cores-1)
    
    w2v_model.build_vocab(w2v_df, progress_per=10000)
    w2v_model.train(w2v_df, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)
    return w2v_model


df[["review"]] = df[["review"]].astype(str)
df["review"] = df["review"].apply(remover)
df["review"] = df["review"].apply(change_lower)
df["review"] = df["review"].apply(clean_data)


w2v_df = get_w2vdf(df)
w2v_model = train_w2v(w2v_df)
w2v_model["film"]
