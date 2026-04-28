import re
import pandas as pd
import nltk
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
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
    
class ResponseEncoding:
    def __init__(self, df):
        self.df = df

    def response_encode(self, col, target):
        mean_target = self.df.groupby(col)[target.name].mean()
        return self.df[col].map(mean_target)

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

class FrequencyEncoding:
    def __init__(self, df):
        self.df = df

    def frequency_encode(self, col):
        freq = self.df[col].value_counts() / len(self.df)
        return self.df[col].map(freq)
    
class HashingEncoding:
    def __init__(self, df):
        self.df = df

    def hashing_encode(self, col, n_features=10):
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        hashed_features = hasher.transform(self.df[col].astype(str))
        return hashed_features


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
    
class FisrtLastText:
    def __init__(self, df):
        self.df = df

    def extract_first_last(self, col):
        self.df[col] = self.df[col].fillna("")
        self.df[f"{col}_first"] = self.df[col].apply(lambda x: x.split()[0] if len(x.split()) > 0 else "")
        self.df[f"{col}_last"] = self.df[col].apply(lambda x: x.split()[-1] if len(x.split()) > 0 else "")
        return self.df[[f"{col}_first", f"{col}_last"]]

class TruncatedText:
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

class Balancing:
    """
        Using SMOTE
    """
    def __init__(self, df):
        self.df = df

    def smote_over_sample(self, X, y):
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def smote_under_sample(self, X, y):
        under_sampler = RandomUnderSampler()
        X_resampled, y_resampled = under_sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def smote_combined_sample(self, X, y):
        smote_enn = SMOTEENN()
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        return X_resampled, y_resampled