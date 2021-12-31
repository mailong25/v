from lipsyn import LipSyner
from flask import Response
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import cv2, time, json
import settings, uuid, redis

db = redis.StrictRedis(host=settings.REDIS_HOST, password = settings.REDIS_PASS,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

FRAME_BATCH_SIZE = 20
RESOURCE_PATH = '/home/ubuntu/resources/Wav2Lip/'

def multi_pop(r, q, n):
    p = r.pipeline()
    p.multi()
    p.lrange(q, 0, n - 1)
    p.ltrim(q, n, -1)
    return p.execute()[0]

print(multi_pop(db,settings.CONTEXT_QUEUE,settings.BATCH_SIZE))

if __name__ == "__main__":
    
    model = LipSyner(RESOURCE_PATH + 'wav2lip_gan_av.pth', RESOURCE_PATH + 'model_480.avi',  RESOURCE_PATH + 'face_480.pkl')
    
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of items, then initialize the IDs and batch of items themselves
        reqs = multi_pop(db,settings.CONTEXT_QUEUE,settings.BATCH_SIZE)
        
        # loop over the queue
        for req in reqs:
            req = json.loads(req)
            check = False
            frame_batch = []
            cur_idx = 0
            
            for frame in model.predict(req["audio_path"],req["start_index"]):
                frame_batch.append(cv2.imencode('.jpg',frame)[1].tostring())
                if len(frame_batch) >= FRAME_BATCH_SIZE:
                    
                    if check == False:
                        db.set(req["id"] + '-' + str(cur_idx), b'endofframe'.join(frame_batch))
                        check = True
                    else:
                        db.set(req["id"] + '-' + str(cur_idx), b'endofframe' + b'endofframe'.join(frame_batch))

                    frame_batch = []
                    cur_idx += 1
            
            if len(frame_batch) > 0:
                db.set(req["id"] + '-' + str(cur_idx), b'endofframe'.join(frame_batch))
                
            db.set(req["id"] + '-' + 'end', b'')
        
        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)
