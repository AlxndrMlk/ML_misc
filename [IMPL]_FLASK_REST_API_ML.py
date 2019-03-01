
# coding: utf-8

import pickle

from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

# Example request:
# curl -X POST http://127.0.0.1:5000/ -d '{'text': "SOME TEXT HERE"}' 
# -H 'Content-Type: application/json'

@app.route('/', methods=['GET', 'POST'])
def predict():
    text = request.get_json()['text']
    pred_class = str(pipe.predict([text])[0])
    return jsonify({'prediction': pred_class})

if __name__ == '__main__':
    with open('clf_news_pipe.pkl', 'rb') as file:
        pipe = pickle.load(file)
        
    app.run(port=5000, debug=True)
