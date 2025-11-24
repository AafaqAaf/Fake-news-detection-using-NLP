import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("Loading datasets...")
try:
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')
except FileNotFoundError:
    print("Error: CSV files not found. Please ensure 'True.csv' and 'Fake.csv' are in the same directory.")
    exit()

# Add labels: 1 for True, 0 for Fake
true_df['label'] = 1
fake_df['label'] = 0

# Merge datasets
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total records: {df.shape[0]}")

# Preprocessing
print("Preprocessing text...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove non-alphabetic characters and lower case
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    # Tokenize and remove stopwords & lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Feature Engineering
print("Vectorizing text...")
X = df['clean_text']
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Model Training
print("Training models...")
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    print(f"{name} trained.")

# Save models and vectorizer
print("Saving artifacts...")
joblib.dump(models, 'models.joblib')
joblib.dump(tfidf, 'vectorizer.joblib')
print("Done! 'models.joblib' and 'vectorizer.joblib' saved.")
