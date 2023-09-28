# generate_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import re

def preprocess_text(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load the dataset
df = pd.read_csv('WELFake_Dataset.csv')

# Handling missing values by filling NaN with a placeholder
df['title'].fillna('NA', inplace=True)
df['text'].fillna('NA', inplace=True)

# Combine "title" and "text" columns into a single input feature
X = df['title'] + ' ' + df['text']
y = df['label']

# Apply text preprocessing
X = X.apply(preprocess_text)

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Initialize and train the PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Evaluate the model on the test data
tfidf_test = tfidf_vectorizer.transform(X_test)
y_pred = pac.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Model Accuracy: {accuracy}')

# Save the model and TF-IDF vectorizer to files
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(pac, 'fake_news_model.pkl')
