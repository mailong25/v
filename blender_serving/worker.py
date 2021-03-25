# import the necessary packages
import numpy as np
import settings
import redis
import time
import json
from blender import Blender
import logging
#logging.basicConfig(filename='worker.log', level=logging.INFO)

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def multi_pop(r, q, n):
    p = r.pipeline()
    p.multi()
    p.lrange(q, 0, n - 1)
    p.ltrim(q, n, -1)
    return p.execute()[0]

print(multi_pop(db,settings.CONTEXT_QUEUE,settings.BATCH_SIZE))

def classify_process():
    
    model = Blender(model_file = '/home/ubuntu/blender/ParlAI/data/models/blender/blender_1Bdistill/model',
                    parlai_home = '/home/ubuntu/blender/ParlAI',include_personas = True)
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of items, then initialize the IDs and batch of items themselves
        queue = multi_pop(db,settings.CONTEXT_QUEUE,settings.BATCH_SIZE)
        ids = []
        contexts = []
        responses = []
        
        # loop over the queue
        for q in queue:
            q = json.loads(q)
            contexts.append(q["context"])
            ids.append(q["id"])
        
        if len(ids) > 0:
            responses = [model.predict(c) for c in contexts]
            for (id_, res) in zip(ids, responses):
                db.set(id_, json.dumps(res))

#             # remove the set of images from our queue
#             db.ltrim(settings.CONTEXT_QUEUE, len(ids), -1)

        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    classify_process()
