import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

stop_words = set(stopwords.words('english'))


# Text Cleaning
def clean_text(text):
    if isinstance(text, str):
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return ' '.join([w for w in text.split() if w not in stop_words])
    return ""


# Load + Merge Data
def load_data(variant_path, text_path):
    data = pd.read_csv(variant_path)
    data_text = pd.read_csv(
        text_path,
        sep='\t',
        names=["ID", "TEXT"],
        skiprows=1
    )

    df = pd.concat([data, data_text], axis=1)
    df.loc[df['TEXT'].isnull(), 'TEXT'] = df['Gene'] + ' ' + df['Variation']

    df['Class'] = df['Class'].astype(int)
    return df


# Split Data
def split_data(df):
    y = df['Class'].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y, stratify=y, test_size=0.36, random_state=42
    )

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.5555, random_state=42
    )

    return X_train, X_cv, X_test, y_train, y_cv, y_test


# Apply Text Cleaning
def apply_text_cleaning(train_df, cv_df, test_df):
    for df in [train_df, cv_df, test_df]:
        df['TEXT'] = df['TEXT'].apply(clean_text)

    return train_df, cv_df, test_df


# TF-IDF Feature Engineering
def vectorize(train_df, cv_df, test_df):
    tfidf_text = TfidfVectorizer(min_df=3, max_features=10000)
    tfidf_gene = TfidfVectorizer()
    tfidf_var = TfidfVectorizer()

    train_text = tfidf_text.fit_transform(train_df['TEXT'])
    cv_text = tfidf_text.transform(cv_df['TEXT'])
    test_text = tfidf_text.transform(test_df['TEXT'])

    train_gene = tfidf_gene.fit_transform(train_df['Gene'])
    cv_gene = tfidf_gene.transform(cv_df['Gene'])
    test_gene = tfidf_gene.transform(test_df['Gene'])

    train_var = tfidf_var.fit_transform(train_df['Variation'])
    cv_var = tfidf_var.transform(cv_df['Variation'])
    test_var = tfidf_var.transform(test_df['Variation'])

    train_x = hstack([train_gene, train_var, train_text]).tocsr()
    cv_x = hstack([cv_gene, cv_var, cv_text]).tocsr()
    test_x = hstack([test_gene, test_var, test_text]).tocsr()

    vectorizers = {
        "text": tfidf_text,
        "gene": tfidf_gene,
        "variation": tfidf_var
    }

    return train_x, cv_x, test_x, vectorizers