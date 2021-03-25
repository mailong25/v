import numpy as np
import settings
import flask
import redis
import uuid
import time
import json
import io
import logging
#logging.basicConfig(filename='web.log', level=logging.INFO)

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

@app.route("/")
def homepage():
    return "Welcome to the REST API!"

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    context = flask.request.json['context']
    id_ = str(uuid.uuid4())
    db.rpush(settings.CONTEXT_QUEUE, json.dumps({"id": id_, "context": context}))
    
    # keep looping until our model server returns the output predictions
    while True:
        # attempt to grab the output predictions
        output = db.get(id_)
        # check to see if our model has classified the input
        if output is not None:
            data["response"] = json.loads(output.decode("utf-8"))
            db.delete(id_)
            break

        # sleep for a small amount to give the model a chance to predict
        time.sleep(settings.CLIENT_SLEEP)
    
    data["success"] = True
    return flask.jsonify(data)

if __name__ == "__main__":
    print("* Starting web service...")
    app.run(port=5901)
