

import pandas as pd
import re
import contractions
from nltk.corpus import stopwords
from gensim.models import Word2Vec,KeyedVectors
import multiprocessing
from bs4 import BeautifulSoup as bs
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
    
df = pd.read_csv('/Users/ysmn/Desktop/digitastic/word2vec/labeledTrainData.tsv', header=0, delimiter=r'\t', quoting=3)
df.head()  




df["review"]=df['review'].apply(lambda x: [contractions.fix(word) for word in x.split()])
# stop = stopwords.words('english')
# df['clean_review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df[["review"]] = df[["review"]].astype(str)
stopwords=stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()

def cleaning(raw_text):
    # Removing HTML Tags
    html_removed_text=bs(raw_text).get_text()
    
    # Remove any non character
    character_only_text=re.sub("[^a-zA-Z]"," ",html_removed_text)
    
    # Lowercase and split
    lower_text=character_only_text.lower().split()
    
    # Get STOPWORDS and remove
    stop_remove_text=[i for i in lower_text if not i in stopwords]
    
    # Remove one character words
    lemma_removed_text=[word for word in stop_remove_text if len(word)>1]
    
    #Lemmatization
    lemma_removed_text=[wordnet_lemmatizer.lemmatize(word,'v') for word in stop_remove_text]
    
    
    return " ".join(lemma_removed_text) 


df['clean_review']=df['review'].apply(cleaning)

df['clean_review'][5]

# def clean_data(text):
#     text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
#     text = re.sub(r'[\\/×\^\]\[÷]', '', text)
#     return text

# def change_lower(text):
#     text = text.lower()
#     return text

# def remover(text):
#     stop_words= list(stopwords.words('english'))
#     text_tokens = text.split(" ")
#     final_list = [word for word in text_tokens if not word in stop_words]
#     text = ' '.join(final_list)
#     return text

def get_w2vdf(df):
    w2v_df = pd.DataFrame(df["clean_review"]).values.tolist()
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
                         sg = 0,
                         workers=cores-1)
    
    w2v_model.build_vocab(w2v_df, progress_per=10000)
    w2v_model.train(w2v_df, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)
    return w2v_model


df[["clean_review"]] = df[["clean_review"]].astype(str)
# df["clean_review"] = df["clean_review"].apply(remover)
# df["clean_review"] = df["clean_review"].apply(change_lower)
# df["clean_review"] = df["clean_review"].apply(clean_data)



w2v_df = get_w2vdf(df)
w2v_model = train_w2v(w2v_df)
w2v_model.wv["film"].most_similar
w2v_model.wv.most_similar(positive=["film","watch"], topn=5)

w2v_model.wv.save_word2vec_format('/Users/ysmn/Desktop/digitastic/model/gensim_w2v_model.bin',binary=True)
# trained_model= KeyedVectors.load_word2vec_format('/Users/ysmn/Desktop/digitastic/model/gensim_w2v_model.bin', binary=True)
