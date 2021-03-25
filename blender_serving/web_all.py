import numpy as np
import settings
import flask
#import redis
import uuid
import time
import json
import io
import logging
#logging.basicConfig(filename='web.log', level=logging.INFO)
# initialize our Flask application and Redis server
from blender import Blender

app = flask.Flask(__name__)

PAR_PATH = "/home/ubuntu/blender/ParlAI"
model = Blender(model_file = PAR_PATH + '/data/models/blender/blender_1Bdistill/model',
            parlai_home = PAR_PATH,include_personas = True)
print("* Model loaded")

@app.route("/")
def homepage():
    return "Welcome to the REST API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    context = flask.request.json['context']
    print(context)
    data["success"] = True
    data["response"] = model.predict(context)
    return flask.jsonify(data)

if __name__ == "__main__":
    print("* Starting web service...")
    app.run(host='127.0.0.1', port=5901,debug=False)
