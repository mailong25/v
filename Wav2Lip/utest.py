import time, requests
from threading import Thread
from random import randint
time.sleep(5)
import os
AUDIO_URL = 'test_wavs/'
paths = [AUDIO_URL + p for p in os.listdir(AUDIO_URL)]
paths = sorted(list(set(paths)))

LIPSYNC_URL = 'http://0.0.0.0:5800/lip'

def lip_req(path):
    start = time.time()
    z = 0 
    with requests.post(LIPSYNC_URL, json = {'audio_path': path,'start_index': randint(100,1000)}, 
                       stream = True) as r:
        
        for chunk in r.iter_content(chunk_size=200000):
            frames = chunk.split(b'endofframe')
            #if z % 20 == 0:
            if z == 0:
                print(path, time.time() - start)
                
            z += 1
    print(path, time.time() - start)

for i in range(0,16):
    Thread(target = lip_req, args = (paths[i],), daemon=False).start()
    #time.sleep(0.2)
