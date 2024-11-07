import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Ensure nltk downloads are present
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
wordnet = WordNetLemmatizer()

# Load and preprocess data
df = pd.read_csv('spam.csv', encoding='latin-1')
df.dropna(how='any', axis=1, inplace=True)
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean message text
def cleaner(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    text = ' '.join([wordnet.lemmatize(word) for word in text.split() if not word in stop_words])
    return text

df['clean_msg'] = df['message'].apply(cleaner)

# Feature extraction
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['clean_msg'].to_numpy())
y = df['label'].values

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train models
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)
svm.fit(x_train, y_train)

nb_classifier = MultinomialNB()
nb_classifier.fit(x_train, y_train)

# Function to classify messages
def classify_message(message, model):
    message_clean = cleaner(message)
    message_vector = vectorizer.transform([message_clean])
    prediction = model.predict(message_vector)[0]
    return 'Spam' if prediction == 1 else 'Not Spam'

# GUI Setup
def check_spam():
    message = entry_message.get()
    if not message:
        messagebox.showwarning("Input Error", "Please enter a message to check.")
        return

    result_svm = classify_message(message, svm)
    result_nb = classify_message(message, nb_classifier)

    result_text = f"SVM Prediction: {result_svm}\nNaive Bayes Prediction: {result_nb}"
    messagebox.showinfo("Prediction Result", result_text)

# Create the main window
window = tk.Tk()
window.title("SpamShield")
window.geometry("400x200")

# Add GUI components
label_message = tk.Label(window, text="Enter a message:")
label_message.pack(pady=5)

entry_message = tk.Entry(window, width=50)
entry_message.pack(pady=5)

button_check = tk.Button(window, text="Check Spam", command=check_spam)
button_check.pack(pady=10)

# Run the main loop
window.mainloop()
