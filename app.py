import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form['comment']
    prediction = model.predict(int_features)

    return render_template('index.html', prediction_text='percentage of toxic content in the comment {}':, prediction)


if __name__ == "__main__":
    app.run(debug=True)
