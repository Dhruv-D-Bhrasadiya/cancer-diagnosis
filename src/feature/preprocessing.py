import re
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
from category_encoders import BinaryEncoder, HashingEncoder, TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# ===================== SETUP =====================
# Ensure stopwords are available
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))

try:
    from gensim.models import Word2Vec
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("[WARNING] Word2Vec not available. Install gensim to use Word2Vec features.")


# ===================== TEXT CLEANING =====================
def clean_text(text):
    """Basic text cleaning: lowercase, remove special chars, remove stopwords."""
    if isinstance(text, str):
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return ' '.join([w for w in text.split() if w not in STOP_WORDS])
    return ""


def apply_text_cleaning(df):
    """Apply text cleaning to all text columns."""
    df = df.copy()
    
    df['TEXT'] = df['TEXT'].fillna("")
    df['Gene'] = df['Gene'].fillna("Unknown")
    df['Variation'] = df['Variation'].fillna("Unknown")
    
    df['TEXT'] = df['TEXT'].apply(clean_text)
    
    return df


# ===================== DATA SPLITTING =====================
def split_data(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Split data into train/validation/test sets with stratification.
    Labels are converted from 1-9 to 0-8.
    """
    df = df.copy()
    
    # Convert labels from 1-9 to 0-8
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(int) - 1
        y = df['Class'].values
    else:
        raise ValueError("DataFrame must contain 'Class' column")
    
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        df, y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split: separate train and validation
    relative_val_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        stratify=y_temp,
        test_size=relative_val_size,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ===================== CATEGORICAL FEATURE ENCODING =====================

class CategoricalEncoder:
    """Wrapper class for different categorical encoding techniques."""
    
    def __init__(self, method="tfidf", target_col="Gene"):
        """
        Args:
            method: Encoding method ('response', 'target', 'frequency', 'hashing', 'binary', 'tfidf')
            target_col: Column to encode ('Gene' or 'Variation')
        """
        self.method = method
        self.target_col = target_col
        self.encoder = None
        self.mapping = None
        
    def fit(self, X_train, y_train):
        """Fit the encoder on training data."""
        if self.method == "response":
            self._fit_response_coding(X_train, y_train)
        elif self.method == "target":
            self._fit_target_encoding(X_train, y_train)
        elif self.method == "frequency":
            self._fit_frequency_encoding(X_train, y_train)
        elif self.method == "binary":
            self._fit_binary_encoding(X_train, y_train)
        elif self.method == "hashing":
            # Hashing doesn't need fitting, just convert to strings
            pass
        elif self.method == "tfidf":
            self.encoder = TfidfVectorizer(max_features=100)
            self.encoder.fit(X_train[self.target_col].astype(str))
        return self
    
    def transform(self, X):
        """Transform data using fitted encoder."""
        if self.method == "response":
            return self._transform_response_coding(X)
        elif self.method == "target":
            return self._transform_target_encoding(X)
        elif self.method == "frequency":
            return self._transform_frequency_encoding(X)
        elif self.method == "binary":
            return self._transform_binary_encoding(X)
        elif self.method == "hashing":
            return self._transform_hashing(X)
        elif self.method == "tfidf":
            return self.encoder.transform(X[self.target_col].astype(str))
    
    def _fit_response_coding(self, X_train, y_train):
        """Response Coding: encode as mean target value."""
        self.mapping = X_train.groupby(self.target_col).apply(
            lambda x: y_train[x.index].mean()
        ).to_dict()
    
    def _transform_response_coding(self, X):
        """Transform using response coding."""
        encoded = X[self.target_col].map(self.mapping).fillna(0).values
        return csr_matrix(encoded.reshape(-1, 1))
    
    def _fit_target_encoding(self, X_train, y_train):
        """Target Encoding using category_encoders."""
        encoder = TargetEncoder(cols=[self.target_col])
        X_copy = X_train.copy()
        encoder.fit(X_copy, y_train)
        self.encoder = encoder
    
    def _transform_target_encoding(self, X):
        """Transform using target encoding."""
        X_copy = X.copy()
        encoded = self.encoder.transform(X_copy)
        return csr_matrix(encoded[[self.target_col]].values)
    
    def _fit_frequency_encoding(self, X_train, y_train):
        """Frequency Encoding: encode as count/frequency."""
        freq = X_train[self.target_col].value_counts().to_dict()
        total = len(X_train)
        self.mapping = {k: v / total for k, v in freq.items()}
    
    def _transform_frequency_encoding(self, X):
        """Transform using frequency encoding."""
        encoded = X[self.target_col].map(self.mapping).fillna(0).values
        return csr_matrix(encoded.reshape(-1, 1))
    
    def _fit_binary_encoding(self, X_train, y_train):
        """Binary Encoding using category_encoders."""
        encoder = BinaryEncoder(cols=[self.target_col])
        X_copy = X_train.copy()
        encoder.fit(X_copy)
        self.encoder = encoder
    
    def _transform_binary_encoding(self, X):
        """Transform using binary encoding."""
        X_copy = X.copy()
        encoded = self.encoder.transform(X_copy)
        return encoded.astype(float).values
    
    def _transform_hashing(self, X):
        """Hashing Encoding using HashingEncoder."""
        encoder = HashingEncoder(cols=[self.target_col], n_components=64)
        X_copy = X.copy()
        encoded = encoder.transform(X_copy)
        return encoded.astype(float).values


# ===================== TEXT FEATURE EXTRACTION =====================

class TextFeatureExtractor:
    """Wrapper class for different text preprocessing techniques."""
    
    def __init__(self, method="tfidf", max_features=1000):
        """
        Args:
            method: Text preprocessing method ('tfidf', 'word2vec', 'truncated', 'first_last')
            max_features: Maximum features for TF-IDF
        """
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.w2v_model = None
        
    def fit(self, texts):
        """Fit the text feature extractor on training texts."""
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.9
            )
            self.vectorizer.fit(texts)
        elif self.method == "word2vec":
            if not WORD2VEC_AVAILABLE:
                raise ImportError("Word2Vec requires gensim. Install with: pip install gensim")
            # Tokenize texts
            tokenized = [text.split() for text in texts]
            self.w2v_model = Word2Vec(
                sentences=tokenized,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4
            )
        return self
    
    def transform(self, texts):
        """Transform texts using fitted extractor."""
        if self.method == "tfidf":
            return self.vectorizer.transform(texts)
        elif self.method == "word2vec":
            return self._transform_word2vec(texts)
        elif self.method == "truncated":
            return self._transform_truncated(texts)
        elif self.method == "first_last":
            return self._transform_first_last(texts)
    
    def _transform_word2vec(self, texts):
        """Average Word2Vec embeddings for each text."""
        vectors = []
        for text in texts:
            words = text.split()
            word_vecs = [self.w2v_model.wv[w] for w in words if w in self.w2v_model.wv]
            if word_vecs:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                vectors.append(np.zeros(self.w2v_model.vector_size))
        return csr_matrix(np.array(vectors))
    
    def _transform_truncated(self, texts, max_length=100):
        """Keep only first N words of each text."""
        truncated = [' '.join(text.split()[:max_length]) for text in texts]
        vectorizer = TfidfVectorizer(max_features=self.max_features, min_df=2)
        vectorizer.fit(truncated)
        return vectorizer.transform(truncated)
    
    def _transform_first_last(self, texts, n_words=50):
        """Use first N and last N words of each text."""
        combined = []
        for text in texts:
            words = text.split()
            first_n = ' '.join(words[:n_words])
            last_n = ' '.join(words[-n_words:])
            combined.append(first_n + ' ' + last_n)
        vectorizer = TfidfVectorizer(max_features=self.max_features, min_df=2)
        vectorizer.fit(combined)
        return vectorizer.transform(combined)


# ===================== DATA BALANCING =====================

class BalancingTechnique:
    """Wrapper class for different balancing techniques."""
    
    def __init__(self, method="none", random_state=42):
        """
        Args:
            method: Balancing method ('none', 'smote_oversample', 'smote_undersample', 'smote_hybrid')
            random_state: Random state for reproducibility
        """
        self.method = method
        self.random_state = random_state
        self.pipeline = None
        
    def fit_resample(self, X, y):
        """Apply balancing to training data."""
        if self.method == "none":
            return X, y
        elif self.method == "smote_oversample":
            return self._apply_smote_oversample(X, y)
        elif self.method == "smote_undersample":
            return self._apply_smote_undersample(X, y)
        elif self.method == "smote_hybrid":
            return self._apply_smote_hybrid(X, y)
    
    def _apply_smote_oversample(self, X, y):
        """SMOTE with oversampling only."""
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        return smote.fit_resample(X, y)
    
    def _apply_smote_undersample(self, X, y):
        """SMOTE with undersampling only."""
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        under = RandomUnderSampler(random_state=self.random_state)
        return under.fit_resample(X_smote, y_smote)
    
    def _apply_smote_hybrid(self, X, y):
        """SMOTE hybrid: combine over and undersampling."""
        pipeline = ImbPipeline([
            ('over', SMOTE(random_state=self.random_state, k_neighbors=5)),
            ('under', RandomUnderSampler(random_state=self.random_state))
        ])
        return pipeline.fit_resample(X, y)


# ===================== PREPROCESSING PIPELINE =====================

class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    def __init__(
        self,
        categorical_encoding="tfidf",  # Method for Gene/Variation
        categorical_target="Gene",      # Which column to encode: Gene, Variation, or both
        text_method="tfidf",            # Text preprocessing method
        balancing_method="none",        # Balancing technique
        max_text_features=1000,
        random_state=42
    ):
        self.categorical_encoding = categorical_encoding
        self.categorical_target = categorical_target
        self.text_method = text_method
        self.balancing_method = balancing_method
        self.max_text_features = max_text_features
        self.random_state = random_state
    
    def __repr__(self):
        return (f"Config(cat_enc={self.categorical_encoding}, "
                f"cat_target={self.categorical_target}, "
                f"text={self.text_method}, "
                f"balance={self.balancing_method})")


def preprocess_pipeline(
    train_df,
    val_df,
    test_df,
    config=None,
    **kwargs
):
    """
    End-to-end preprocessing pipeline with configurable techniques.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        config: PreprocessingConfig object or dict of kwargs
        **kwargs: Alternative way to pass config parameters
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Parse config
    if config is None:
        config = PreprocessingConfig(**kwargs)
    elif isinstance(config, dict):
        config = PreprocessingConfig(**config)
    
    print(f"[INFO] Using preprocessing config: {config}")
    
    # 1. Clean text
    print("[INFO] Cleaning text...")
    train_df = apply_text_cleaning(train_df)
    val_df = apply_text_cleaning(val_df)
    test_df = apply_text_cleaning(test_df)
    
    # 2. Extract labels and remove Class column from features
    y_train = (train_df['Class'].astype(int) - 1).values
    y_val = (val_df['Class'].astype(int) - 1).values
    y_test = (test_df['Class'].astype(int) - 1).values
    
    # 3. Encode categorical features (Gene and/or Variation)
    print("[INFO] Encoding categorical features...")
    categorical_parts = []
    
    if config.categorical_target in ["Gene", "both"]:
        gene_encoder = CategoricalEncoder(config.categorical_encoding, "Gene")
        gene_encoder.fit(train_df, y_train)
        X_train_gene = gene_encoder.transform(train_df)
        X_val_gene = gene_encoder.transform(val_df)
        X_test_gene = gene_encoder.transform(test_df)
        categorical_parts.append((X_train_gene, X_val_gene, X_test_gene))
    
    if config.categorical_target in ["Variation", "both"]:
        var_encoder = CategoricalEncoder(config.categorical_encoding, "Variation")
        var_encoder.fit(train_df, y_train)
        X_train_var = var_encoder.transform(train_df)
        X_val_var = var_encoder.transform(val_df)
        X_test_var = var_encoder.transform(test_df)
        categorical_parts.append((X_train_var, X_val_var, X_test_var))
    
    # 4. Extract text features
    print("[INFO] Extracting text features...")
    text_extractor = TextFeatureExtractor(config.text_method, config.max_text_features)
    text_extractor.fit(train_df['TEXT'].values)
    X_train_text = text_extractor.transform(train_df['TEXT'].values)
    X_val_text = text_extractor.transform(val_df['TEXT'].values)
    X_test_text = text_extractor.transform(test_df['TEXT'].values)
    
    # 5. Combine all features
    print("[INFO] Combining features...")
    features_to_combine = [X_train_text, X_val_text, X_test_text]
    for X_train_cat, X_val_cat, X_test_cat in categorical_parts:
        features_to_combine.extend([X_train_cat, X_val_cat, X_test_cat])
    
    X_train = hstack([features_to_combine[0]] + [f for i, f in enumerate(features_to_combine) if i % 3 == 1]).tocsr()
    X_val = hstack([features_to_combine[1]] + [f for i, f in enumerate(features_to_combine) if i % 3 == 2]).tocsr()
    X_test = hstack([features_to_combine[2]] + [f for i, f in enumerate(features_to_combine) if i % 3 == 0]).tocsr()
    
    # Simplified combination (fix the above)
    train_parts = [X_train_text]
    val_parts = [X_val_text]
    test_parts = [X_test_text]
    
    for X_train_cat, X_val_cat, X_test_cat in categorical_parts:
        train_parts.append(X_train_cat)
        val_parts.append(X_val_cat)
        test_parts.append(X_test_cat)
    
    X_train = hstack(train_parts).tocsr()
    X_val = hstack(val_parts).tocsr()
    X_test = hstack(test_parts).tocsr()
    
    # 6. Apply balancing technique
    print("[INFO] Applying balancing technique...")
    balancer = BalancingTechnique(config.balancing_method, config.random_state)
    X_train, y_train = balancer.fit_resample(X_train, y_train)
    
    print(f"[INFO] Final shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"[INFO] Class distribution - Train: {np.bincount(y_train)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_all_preprocessing_configs():
    """Generate all permutations of preprocessing configurations."""
    categorical_encodings = ["tfidf", "response", "target", "frequency", "hashing", "binary"]
    categorical_targets = ["Gene", "Variation", "both"]
    text_methods = ["tfidf", "truncated", "first_last"]
    if WORD2VEC_AVAILABLE:
        text_methods.append("word2vec")
    balancing_methods = ["none", "smote_oversample", "smote_undersample", "smote_hybrid"]
    
    configs = []
    for cat_enc in categorical_encodings:
        for cat_target in categorical_targets:
            for text_method in text_methods:
                for balance in balancing_methods:
                    config = PreprocessingConfig(
                        categorical_encoding=cat_enc,
                        categorical_target=cat_target,
                        text_method=text_method,
                        balancing_method=balance
                    )
                    configs.append(config)
    
    return configs


"""
How to use it:

# Method 1: Use default config
from src.data.loader import load_processed_data
from src.feature.preprocessing import preprocess_pipeline

train_df, val_df, test_df, _ = load_processed_data()
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(train_df, val_df, test_df)

# Method 2: Use custom config
from src.feature.preprocessing import PreprocessingConfig, preprocess_pipeline

config = PreprocessingConfig(
    categorical_encoding="target",
    categorical_target="both",
    text_method="word2vec",
    balancing_method="smote_hybrid"
)
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(
    train_df, val_df, test_df, 
    config=config
)

# Method 3: Generate and iterate through all configs
from src.feature.preprocessing import get_all_preprocessing_configs, preprocess_pipeline

configs = get_all_preprocessing_configs()
for config in configs:
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(
        train_df, val_df, test_df,
        config=config
    )
    # Train model with this config
    ...
"""