
# coding: utf-8

# In[1]:


import pickle

from flask import Flask
from flask import request, jsonify


# In[2]:


app = Flask(__name__)

# Example request:
# curl -X POST http://127.0.0.1:5000/ -d '{'text': "SOME TEXT HERE"}' 
# -H 'Content-Type: application/json'


# In[4]:


@app.route('/', methods=['GET', 'POST'])
def predict():
    text = request.get_json()['text']
    pred_class = str(pipe.predict([text])[0])
    return jsonify({'prediction': pred_class})


# In[6]:


if __name__ == '__main__':
    with open('clf_news_pipe.pkl', 'rb') as file:
        pipe = pickle.load(file)
        
    app.run(port=5000, debug=True)

