import pickle
from flask import Flask, request, app, url_for,render_template
from numpy import array
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

app = Flask(__name__)

classifier = pickle.load(open(r"weights/model_weight.pkl", "rb"))
selector = pickle.load(open(r"weights/selector_weight.pkl", "rb"))
count_vectoriser = pickle.load(open(r"weights/count_vectoriser_weight.pkl", "rb"))

def porter_stemming_message(message:str)->list:
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # review = ' '.join(review)
    return ' '.join(review)
    
    # return list(review)

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/predict")
def load():
    return render_template("/predict.html")

@app.route("/predict", methods=["POST"])

def predict():
    message = request.form.get("message")

    processed_message = porter_stemming_message(message)
    
    X = count_vectoriser.transform([processed_message]).toarray()
    X = selector.transform(X)
    
    prediction = "Spam" if classifier.predict(X)[0] else "Ham"
    
    return render_template("predict.html", prediction_result = f"The message is {prediction}")

if __name__ == "__main__":
    app.run(debug=True)