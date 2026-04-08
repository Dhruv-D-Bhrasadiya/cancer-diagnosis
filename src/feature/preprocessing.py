import re
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


# Global Stopwords

# Ensure stopwords are available
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))


# Text Cleaning
def clean_text(text):
    if isinstance(text, str):
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return ' '.join([w for w in text.split() if w not in STOP_WORDS])
    return ""


# Apply Cleaning
def apply_text_cleaning(df):
    df = df.copy()

    df['TEXT'] = df['TEXT'].fillna("")
    df['Gene'] = df['Gene'].fillna("Unknown")
    df['Variation'] = df['Variation'].fillna("Unknown")

    df['TEXT'] = df['TEXT'].apply(clean_text)

    return df


# Train / CV / Test Split
def split_data(df, test_size=0.2, cv_size=0.2, random_state=42):
    df = df.copy()

    # 🔧 FIX: convert labels from 1–9 → 0–8
    df['Class'] = df['Class'].astype(int) - 1

    y = df['Class'].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y,
        stratify=y,
        test_size=(test_size + cv_size),
        random_state=random_state
    )

    relative_cv_size = cv_size / (test_size + cv_size)

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp,
        stratify=y_temp,
        test_size=(1 - relative_cv_size),
        random_state=random_state
    )

    return X_train, X_cv, X_test, y_train, y_cv, y_test


# Vectorization
def vectorize(train_df, cv_df, test_df, max_text_features=10000):
    # TF-IDF for each feature type
    tfidf_text = TfidfVectorizer(
        min_df=3,
        max_features=max_text_features
    )

    tfidf_gene = TfidfVectorizer()
    tfidf_var = TfidfVectorizer()

    # TEXT
    train_text = tfidf_text.fit_transform(train_df['TEXT'])
    cv_text = tfidf_text.transform(cv_df['TEXT'])
    test_text = tfidf_text.transform(test_df['TEXT'])

    # Gene
    train_gene = tfidf_gene.fit_transform(train_df['Gene'])
    cv_gene = tfidf_gene.transform(cv_df['Gene'])
    test_gene = tfidf_gene.transform(test_df['Gene'])

    # Variation
    train_var = tfidf_var.fit_transform(train_df['Variation'])
    cv_var = tfidf_var.transform(cv_df['Variation'])
    test_var = tfidf_var.transform(test_df['Variation'])

    # Combine all features
    X_train = hstack([train_gene, train_var, train_text]).tocsr()
    X_cv = hstack([cv_gene, cv_var, cv_text]).tocsr()
    X_test = hstack([test_gene, test_var, test_text]).tocsr()

    vectorizers = {
        "text": tfidf_text,
        "gene": tfidf_gene,
        "variation": tfidf_var
    }

    return X_train, X_cv, X_test, vectorizers


# Full Pipeline Function
def preprocess_pipeline(df, max_text_features=10000):
    """
    End-to-end preprocessing:
    raw df → cleaned → split → vectorized
    """

    # 1. Clean
    df = apply_text_cleaning(df)

    # 2. Split
    train_df, cv_df, test_df, y_train, y_cv, y_test = split_data(df)

    # 3. Vectorize
    X_train, X_cv, X_test, vectorizers = vectorize(
        train_df, cv_df, test_df, max_text_features
    )

    return (
        X_train,
        X_cv,
        X_test,
        y_train,
        y_cv,
        y_test,
        vectorizers
    )


"""
How you use it


from src.data.loader import load_training_data
from src.features.preprocessing import preprocess_pipeline

df = load_training_data()

X_train, X_cv, X_test, y_train, y_cv, y_test, vectorizers = preprocess_pipeline(df)

"""