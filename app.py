from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import time
from tensorflow.keras.models import load_model
import nltk

# Initialize Flask app
app = Flask(__name__)

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

lm = WordNetLemmatizer()

# Load the model from the specified path
model = load_model('D:\DataScience\TWi\Twitter.h5')

def newinput(comment):
    review = re.sub('[^a-zA-Z]', ' ', comment)
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    onehot1 = [one_hot(review, n=50000)]
    encoded_comments1 = pad_sequences(onehot1, maxlen=40)
    
    if model:
        output = model.predict(encoded_comments1)
        output = np.where(output > 0.5, 1, 0)
        if output == 0:
            result = 'negative'
        else:
            result = 'positive'
        return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Utweet = request.form['comment']  # Get input from the form
        result = newinput(Utweet)
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
