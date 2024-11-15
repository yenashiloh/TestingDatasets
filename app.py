from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download the necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('models/logistic_regression_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Define Tagalog stopwords
tagalog_stopwords = {
    "ako", "ikaw", "siya", "kami", "kayo", "sila", "ito", "iyan", "iyan", "mga", "ng", "sa", "para", 
    "ang", "ngunit", "habang", "dahil", "at", "o", "na", "mula", "tungkol", "upang", "hindi", "lahat",
    "isa", "ito", "iyon", "pati", "bukod", "tama", "mali", "wala", "may", "di", "naman"
}

# Combine English and Tagalog stopwords
stopwords = set(ENGLISH_STOP_WORDS).union(tagalog_stopwords)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define words frequently associated with non-cyberbullying contexts
non_cyberbullying_keywords = {"ganda", "bait", "maganda", "friendly", "kamukha", "ligtas"}  # Add more based on your dataset

# Define words associated with cyberbullying contexts (from your dataset)
cyberbullying_keywords = {"bobo", "pangit", "hate", "bully"}  # Example keywords; customize as needed

def preprocess_text(text):
    """Preprocess the input text by cleaning and lemmatizing"""
    # Convert to lowercase
    text = text.lower()
    # Remove unwanted characters
    text = re.sub(r'\[.*?\]', '', text)  # Remove any text inside squ   are brackets
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    
    # Tokenize and remove stopwords, then lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return ' '.join(tokens)

def detect_cyberbullying(text):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    input_tokens = set(processed_text.split())

    # Check if any input tokens match words in texts labeled as cyberbullying
    matched_non_cyberbullying = input_tokens.intersection(non_cyberbullying_keywords)
    matched_cyberbullying = input_tokens.intersection(cyberbullying_keywords)

    # If only non-cyberbullying words are matched, classify as non-cyberbullying
    if matched_non_cyberbullying and not matched_cyberbullying:
        return "No Cyberbullying Detected", 0

    # Otherwise, proceed with model prediction (assuming TF-IDF and model are loaded)
    input_vector = vectorizer.transform([processed_text])
    prediction_prob = model.predict_proba(input_vector)[0][1]  # Probability of class 1 (cyberbullying)
    prediction_label = model.predict(input_vector)[0]

    # Map the prediction to a label (1 = Cyberbullying, 0 = No Cyberbullying)
    result = "Cyberbullying Detected" if prediction_label == 1 else "No Cyberbullying Detected"
    if result == "No Cyberbullying Detected":
        prediction_prob = 0  # Set probability to 0 if no cyberbullying detected

    return result, prediction_prob * 100

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        incident_details = request.form["incident_details"]
        
        # Call the new detection function
        result, probability = detect_cyberbullying(incident_details)
        
        return render_template("index.html", result=result, probability=probability)
    
    return render_template("index.html", result=None, probability=None)

if __name__ == "__main__":
    app.run(debug=True)
