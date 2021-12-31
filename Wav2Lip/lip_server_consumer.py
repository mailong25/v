from lipsyn import LipSyner
from flask import Response
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import cv2, json, time
import settings, uuid, redis
import pickle

db = redis.StrictRedis(host=settings.REDIS_HOST, password = settings.REDIS_PASS,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

app = Flask(__name__)

CORS(app)
@app.route('/lip', methods=['POST'])
def post_lip():
    audio_path   = request.json['audio_path']
    start_index  = request.json['start_index']
    id_ = str(uuid.uuid4())
    
    def generate_frame():        
        db.rpush(settings.CONTEXT_QUEUE, json.dumps({"id": id_, "audio_path": audio_path, "start_index": start_index}))
        cur_idx = 0
        start = time.time()
        
        # keep looping until our model server returns the output predictions
        while True:
            # attempt to grab the output predictions
            res_idx = id_ + '-' + str(cur_idx)
            output = db.get(res_idx)
            # check to see if our model has classified the input
            if output is not None:
                yield output
                db.delete(res_idx)
                cur_idx += 1
                print(time.time() - start)
            elif db.get(id_ + '-' + 'end') is not None:
                break

            # sleep for a small amount to give the model a chance to predict
            time.sleep(settings.CLIENT_SLEEP)
            
    return Response(generate_frame(),content_type = 'application/octet-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5800,debug=False)
