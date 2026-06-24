from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Stemmer object
ps = PorterStemmer()


# Text preprocessing function
def transform_text(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Tokenization
    text = nltk.word_tokenize(text)

    # 3. Remove special characters and punctuation
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    # 4. Remove stopwords
    z = []
    for word in y:
        if word not in stopwords.words('english'):
            z.append(word)

    # 5. Stemming
    final_text = []
    for word in z:
        final_text.append(ps.stem(word))

    return " ".join(final_text)


# Home route
@app.route('/', methods=['GET', 'POST'])
def home():

    result = ""

    if request.method == 'POST':

        message = request.form['message']

        # Preprocess input text
        transformed_sms = transform_text(message)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            result = "Spam Message"
        else:
            result = "Not Spam Message"

    return render_template('index.html', result=result)


# Run application
if __name__ == '__main__':
    app.run(debug=True)