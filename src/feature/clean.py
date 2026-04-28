import re
import pandas as pd
import nltk
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')

try:
    from gensim.models import Word2Vec
except ImportError:
    print("[WARNING] gensim not installed. Word2VecText will not work.")


class OHE:
    def __init__(self, df):
        self.df = df

    def ohe_gene(self, train_df):
        gene_ohe = pd.get_dummies(train_df["Gene"], prefix="Gene")
        return gene_ohe

    def ohe_variation(self, train_df):
        variation_ohe = pd.get_dummies(train_df["Variation"], prefix="Variation")
        return variation_ohe

    def ohe_class(self, train_df):
        class_ohe = pd.get_dummies(train_df["Class"], prefix="Class")
        return class_ohe
    

class TargetEncoding:
    def __init__(self, df):
        self.df = df

    def target_encode(self, col, target, min_samples=100):
        mean = target.mean()
        agg = self.df.groupby(col)[target.name].agg(["mean", "count"])
        smooth = (agg["count"] * agg["mean"] + min_samples * mean) / (agg["count"] + min_samples)
        return self.df[col].map(smooth)
    
class BinaryEncoding:
    def __init__(self, df):
        self.df = df

    def binary_encode(self, col):
        be = BinaryEncoder()
        binary_encoded = be.fit_transform(self.df[[col]])
        return binary_encoded
    
class LabelEncoding:
    def __init__(self, df):
        self.df = df

    def label_encode(self, col):
        le = LabelEncoder()
        label_encoded = le.fit_transform(self.df[col])
        return label_encoded

class TextVectorization:
    def __init__(self, df):
        self.df = df

    def clean_text(self, text):
        """Basic text cleaning: lowercase, remove special chars, remove stopwords."""
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r"[^a-z0-9\s]", " ", text)

            # Add Stopword removal (using nltk as example)
            stop_words = set(stopwords.words('english'))
            text = ' '.join([word for word in text.split() if word not in stop_words])
            
        return text
    
    def tfidf_vectorize(self, col, max_features=5000):
        self.df[col] = self.df[col].fillna("")
        self.df[col] = self.df[col].apply(self.clean_text)
        tfidf = TfidfVectorizer(max_features=max_features)
        text_tfidf = tfidf.fit_transform(self.df[col])
        return text_tfidf
    
class FIRST_LAST_TEXT:
    def __init__(self, df):
        self.df = df

    def extract_first_last(self, col):
        self.df[col] = self.df[col].fillna("")
        self.df[f"{col}_first"] = self.df[col].apply(lambda x: x.split()[0] if len(x.split()) > 0 else "")
        self.df[f"{col}_last"] = self.df[col].apply(lambda x: x.split()[-1] if len(x.split()) > 0 else "")
        return self.df[[f"{col}_first", f"{col}_last"]]

class TRUCATE_TEXT:
    def __init__(self, df):
        self.df = df

    def truncate_text(self, col, max_len=10):
        self.df[col] = self.df[col].fillna("")
        self.df[f"{col}_truncated"] = self.df[col].apply(lambda x: ' '.join(x.split()[:max_len]))
        return self.df[f"{col}_truncated"]
    
class Word2VecText:
    def __init__(self, df):
        self.df = df

    def word2vec_vectorize(self, col, vector_size=100):
        self.df[col] = self.df[col].fillna("")
        sentences = self.df[col].apply(lambda x: x.split()).tolist()
        model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)
        word2vec_features = self.df[col].apply(lambda x: model.wv[x.split()].mean(axis=0) if len(x.split()) > 0 else np.zeros(vector_size))
        return np.vstack(word2vec_features.values)

