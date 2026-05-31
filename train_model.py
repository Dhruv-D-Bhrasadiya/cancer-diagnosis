import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, accuracy_score
from scipy.sparse import hstack
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

project_root = Path('.')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ''
    text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    return ' '.join(w for w in text.split() if w not in stop_words)

print('[INFO] Loading data...')
train_df = pd.read_csv(project_root / 'data/processed/train_data.csv')
val_df   = pd.read_csv(project_root / 'data/processed/val_data.csv')

train_df['TEXT']      = train_df['TEXT'].fillna('').apply(clean_text)
val_df['TEXT']        = val_df['TEXT'].fillna('').apply(clean_text)
train_df['Gene']      = train_df['Gene'].fillna('').str.lower()
val_df['Gene']        = val_df['Gene'].fillna('').str.lower()
train_df['Variation'] = train_df['Variation'].fillna('').str.lower()
val_df['Variation']   = val_df['Variation'].fillna('').str.lower()

print('[INFO] Vectorizing features...')
gene_vec = TfidfVectorizer(max_features=500)
var_vec  = TfidfVectorizer(max_features=500)
text_vec = TfidfVectorizer(max_features=5000)

X_train = hstack([
    gene_vec.fit_transform(train_df['Gene']),
    var_vec.fit_transform(train_df['Variation']),
    normalize(text_vec.fit_transform(train_df['TEXT']), axis=0)
])
X_val = hstack([
    gene_vec.transform(val_df['Gene']),
    var_vec.transform(val_df['Variation']),
    normalize(text_vec.transform(val_df['TEXT']), axis=0)
])

y_train = train_df['Class'].values
y_val   = val_df['Class'].values

print('[INFO] Training Logistic Regression...')
# CalibratedClassifierCV with cv=3 trains + calibrates in one step
model = CalibratedClassifierCV(
    SGDClassifier(class_weight='balanced', alpha=1e-4, penalty='l2',
                  loss='log_loss', random_state=42, max_iter=1000),
    method='sigmoid',
    cv=3
)
model.fit(X_train, y_train)

val_proba = model.predict_proba(X_val)
val_pred  = model.predict(X_val)
print(f'[INFO] Val log-loss : {log_loss(y_val, val_proba):.4f}')
print(f'[INFO] Val accuracy : {accuracy_score(y_val, val_pred):.4f}')

print('[INFO] Saving model and vectorizers...')
out = project_root / 'outputs/models'
out.mkdir(parents=True, exist_ok=True)
joblib.dump(model,    out / 'logreg.joblib')
joblib.dump(gene_vec, out / 'gene_vectorizer.joblib')
joblib.dump(var_vec,  out / 'var_vectorizer.joblib')
joblib.dump(text_vec, out / 'text_vectorizer.joblib')

print('[INFO] Done! Model saved to outputs/models/logreg.joblib')
