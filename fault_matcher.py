import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

df = pd.read_csv("fault_data.csv")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_issue"] = df["issue"].apply(clean_text)
X = df["clean_issue"]
y = df["category"]

# Use cross-validation for small datasets
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())  # Or use LogisticRegression()
])

scores = cross_val_score(model, X, y, cv=min(5, len(df)))
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Optional: Train on all data and save
model.fit(X, y)
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
