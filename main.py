import joblib
import re

def preprocess_text(text):
    # Add text preprocessing steps here (e.g., lowercase, remove punctuation)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load the pre-trained TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
pac = joblib.load('fake_news_model.pkl')

# Get user input for title and text
user_title = input("Enter the news title: ")
user_text = input("Enter the news text: ")

# Combine user input into a single input feature and preprocess it
user_input = preprocess_text(user_title + ' ' + user_text)

# Transform user input using the loaded vectorizer
user_input_tfidf = tfidf_vectorizer.transform([user_input])

# Predict if the news is fake or real and obtain prediction confidence
prediction = pac.predict(user_input_tfidf)
confidence = pac.decision_function(user_input_tfidf)

# Display the result
if prediction[0] == 'fake':
    print(f"This news is likely fake with a confidence score of {confidence[0]:.2f}")
else:
    print(f"This news is likely real with a confidence score of {confidence[0]:.2f}")