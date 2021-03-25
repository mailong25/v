import time, requests
from threading import Thread
from random import randint
time.sleep(8)
import os
texts = ['Hi there , how are you doing',
        'Hi there , how are you doing\nI am fine thank you , what about you\n I am pretty good actually',
        'It is nice to meet you. I am V. How are you doing ?',
        'What do you like to do for fun? I like to play video games.\nHow are you doing\nNo, I do not know that. I do know that I am scared of heights though',
        'Do you know Donald Trump?\nYes, he is the president of the united states\nDo you think he is smart?',
        'I think that he is very smart. He is a businessman and a celebrity',
        'I think so too. He has been a very successful businessman for a very long time now.',
        'Do you know why he lose for this election?\nI am not sure, but I know that he has been very successful\nNo I dont',
        'Me do you have boyfriends?\nI do have a boyfriend. We have been together\n have just had a girlfriend we know through Tinder',
        'I have never used Tinder\nbut it seems like it would be a good way to meet people.\nI think I will give it a try.']

texts = texts + texts + texts + texts + texts

BLENDER_URL = 'http://0.0.0.0:5902/predict'

def blender_req(text):
    start = time.time()
    response = requests.post(BLENDER_URL, json = {"context": text})
    print(time.time() - start)

for i in range(0,10):
    Thread(target = blender_req, args = (texts[i],), daemon=False).start()
    time.sleep(0.01)
