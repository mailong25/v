import numpy as np
import settings
import flask
#import redis
import uuid
import time
import json
import io
import logging
import random
from blender import Blender

app = flask.Flask(__name__)

PAR_PATH = "/home/ubuntu/resources/blender/ParlAI"
model = Blender(model_file = '/home/ubuntu/anaconda3/lib/python3.7/site-packages/data/models/blender/blender_90M/model',
            parlai_home = PAR_PATH,include_personas = False)
print("* Model loaded")

persona = ['your persona: I am a single female',
           'your persona: I am twenty years old',
           'your persona: I am from United States',
           'your persona: I have a brother and a sister',
           'your persona: I am a undergraduate student and I study neuroscience',
           'your persona: I enjoy reading books and watching Youtube videos',
           'your persona: I have a cat and a dog',
           'your persona: I like pop and country music',
           'your persona: I love science fiction and romance movies',]

persona = '\n'.join(persona)

@app.route("/")
def homepage():
    return "Welcome to the REST API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    context = flask.request.json['context']
    context = persona + '\n' + context
    print(context)
    data["success"] = True
    data["response"] = model.predict(context)
    return flask.jsonify(data)

if __name__ == "__main__":
    print("* Starting web service...")
    app.run(host='127.0.0.1', port=5901,debug=False)
