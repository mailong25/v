# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
import cv2
from django.template import loader
import threading
import time 
from random import randint
import base64
from django.http import JsonResponse
from . import global_var
import json
from mutagen.mp3 import MP3
from random import randint
from datetime import datetime, timedelta
from . import tts
import ntpath
import requests
import asyncio
import os
from threading import Thread
from subprocess import Popen, PIPE
import uuid
from os.path import join as join_path
from . import asr
import wave, copy
import logging
logging.basicConfig(filename='log.log', encoding='utf-8', level=logging.INFO)

BLENDER_URL = 'http://35.247.81.213:5901/predict'
LIPSYNC_URL = 'http://54.179.106.170:5800/lip'
SAMPLE_RATE = 44100
MAX_USERS = 2
INACTIVE_THRESHOLD = 20
SILENT_THRESHOLD = 300 * 2
MESSAGE_BUSY = "Thanks for interesting in our service. The server is currently busy. Please try at another time."
MESSAGE_DUP = "You are currently talking to V."
MESSAGE_BACK = 'bot\tIt is great to see you back. What do you wanna talk about today ?'
MESSAGE_START = 'bot\tIt is nice to meet you. I am V. How are you doing ?'

global_frames = global_var.video_frames.get_frames()

users = {}

def user_on_leave(used_id):
    global users
    if used_id in users:
        cur_user = users[used_id]
        with open(cur_user['log_dialog'],'w') as f:
            f.write('\n'.join(cur_user['history']))
        cur_user['asr'].stop()
        del users[used_id]
        logging.info("Remove: " + used_id + " . Users: " + str(users.keys()))

def clean_connection(args):
    global users
    while True:
        cur_time = datetime.now()
        for user_id in list(users.keys()):
            inactive_seconds = (cur_time - users[user_id]['last_update']).total_seconds()
            if inactive_seconds > INACTIVE_THRESHOLD:
                user_on_leave(user_id)
        time.sleep(2)

Thread(target = clean_connection, args = ("clean",)).start()

# Create your views here.
def index(request):
    global users
    if len(users) >= MAX_USERS:
        return HttpResponse(MESSAGE_BUSY)
    
    template = loader.get_template('index.html')
    response = HttpResponse(template.render({}, request))
    user_id = request.COOKIES.get("user_id",None)
    
    if user_id == None:
        user_id = uuid.uuid4().hex
        response.set_cookie("user_id",user_id,10000000,datetime.now() + timedelta(days=3650))
        logging.info("New user_id : " + user_id)
    else:
        logging.info("Existing user_id : " + user_id)

    logging.info(str(users.keys()))
    if user_id in users:
        return HttpResponse(MESSAGE_DUP)
    else:
        return response

def new(request):
    global users
    user_id = request.COOKIES.get("user_id")
    
    users[user_id] = {'last_update': datetime.now(),'cur_frame_id': 0,
                      'lip': {}, 'writer': None, 'tts': tts.TTS(),
                      'asr' : asr.ASR(inactive_mins = 3)}

    cur_user = users[user_id]
    cur_user['log'] = join_path('logs',user_id)
    cur_user['log_dialog'] = join_path(cur_user['log'],'dialog.csv')

    if os.path.exists(cur_user['log']):
        with open(cur_user['log_dialog']) as f:
            turns = f.read().splitlines()
            if turns[-1].split('\t')[0] == 'bot':
                turns[-1] = MESSAGE_BACK
            else:
                turns.append(MESSAGE_BACK)
        cur_user['history'] = turns
    else:
        os.mkdir(cur_user['log'])
        cur_user['history'] = [MESSAGE_START]

    with open(cur_user['log_dialog'],'w') as f:
        f.write('\n'.join(cur_user['history']))
        
    return JsonResponse({})

def stream(user_id):
    global global_frames, users
    logging.info("entering stream!")
    cur_user = users[user_id]
    while (cur_user['cur_frame_id'] < len(global_frames)):
        frame = global_frames[cur_user['cur_frame_id']]
        try:
            frame = cur_user['lip'][cur_user['cur_frame_id']]
        except:
            pass
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)
        
        cur_user['cur_frame_id'] += 1
        
        if cur_user['cur_frame_id'] % 50 == 0:
            cur_user['last_update'] = datetime.now()

def video_feed(request):
    logging.info("Enter video_feed")
    user_id = request.COOKIES['user_id']
    return StreamingHttpResponse(stream(user_id), content_type='multipart/x-mixed-replace; boundary=frame')

def blender_response(dialog_history):
    dialog_history = [t.split('\t')[1] for t in dialog_history]
    context = '\n'.join(dialog_history)
    response = requests.post(BLENDER_URL, json = {"context": context})
    return 'bot\t' + response.json()['response']['text']

def updating_frame(response_audio_wav, start_index, user_id):
    global users
    
    with requests.post(LIPSYNC_URL, json = {'audio_path':response_audio_wav,'start_index': start_index}, 
                       stream = True) as r:

        prev = b''
        frame_count = 0
        for chunk in r.iter_content(chunk_size=200000):
            frames = chunk.split(b'endofframe')
            frames[0] = prev + frames[0]
            for frame in frames[:-1]:
                users[user_id]['lip'][start_index + frame_count] = frame
                frame_count += 1
            prev = frames[-1]

def startfunc(request):
    global users
    user_id = request.COOKIES['user_id']
    cur_user = users[user_id]
    start = time.time()
    user_text = request.POST.get("text")
    if user_text == 'idle':
        response_text = cur_user['history'][-1]
    else:
        cur_user['history'].append('user\t' + user_text)
        response_text = blender_response(cur_user['history'])
        cur_user['history'].append(response_text)
        cur_user['last_update'] = datetime.now()
    
    response_text = response_text.split('\t')[1]
    logging.info("Blender for " + response_text + " is: " + str(time.time() - start))
    response_audio = "/home/ubuntu/chatweb/files/audio/" + uuid.uuid4().hex + ".mp3"
    response_audio_wav = response_audio.replace('.mp3','.wav')
    cur_user['tts'].predict(response_text,response_audio)
    
    convert_cmd = 'ffmpeg -i ' + response_audio + ' ' + response_audio_wav
    Popen(convert_cmd.split(), stdout=PIPE, stderr=PIPE).wait()
    
    audio_len = int(MP3(response_audio).info.length * 1000)
    audio_url = 'https://www.vvreborn.com/files/audio/' + ntpath.basename(response_audio)
    
    logging.info("TTS: " + str(time.time() - start))
    
    del cur_user['lip']
    cur_user['lip'] = {}
    
    start_index = cur_user['cur_frame_id'] + 30
    
    thread = Thread(target = updating_frame, args = (response_audio_wav, start_index, user_id))
    thread.start()
    
    while(len(cur_user['lip']) == 0):
        time.sleep(0.1)
    
    if start_index < cur_user['cur_frame_id']:
        cur_user['cur_frame_id'] = start_index
    
    SLEEP_TIME = float(start_index - cur_user['cur_frame_id']) / 24.0
    SLEEP_TIME = max(0,SLEEP_TIME - 0.2)
    time.sleep(SLEEP_TIME)
    return JsonResponse({'text': response_text,'audio':audio_url, 'audio_len' : audio_len})

def start_reg(request):
    global users
    cur_user = users[request.COOKIES['user_id']]
    cur_user['asr'].start()
    cur_user['writer'] = wave.open(cur_user['log'] + '/' + datetime.now().strftime("%m_%d_%M_%S") + '.wav', 'wb')
    cur_user['writer'].setnchannels(1)
    cur_user['writer'].setsampwidth(2)
    cur_user['writer'].setframerate(SAMPLE_RATE)
    return JsonResponse({})

def stop_reg(request):
    global users
    cur_user = users[request.COOKIES['user_id']]
    cur_user['asr'].stop()
    
    timeout_count = 0 
    while (cur_user['asr'].is_done != True):
        time.sleep(0.2)
        timeout_count += 1
        if timeout_count >= 7:
            print("stop timeout !")
            break
    
    cur_user['writer'].close()
    return JsonResponse({'text': cur_user['asr'].get_text()})

def recognize(request):
    global users
    used_id = request.COOKIES['user_id']
    
    if used_id not in users:
        return JsonResponse({'text': "", 'finish': 'false'})
    
    cur_user = users[used_id]
    data = request.FILES['audio']
    binaryHeader = data.read(44)
    frames = data.read()
    
    if cur_user['asr'].is_inactive():
        user_on_leave(used_id)
        return JsonResponse({'text': "", 'finish': 'close'})
    
    cur_user['asr'].push_frames(frames,SAMPLE_RATE)
    
    try:
        cur_user['writer'].writeframes(frames)
    except:
        pass
    
    return JsonResponse({'text': cur_user['asr'].get_text(), 'finish': 'false'})

def end(request):
    user_on_leave(request.COOKIES['user_id'])
    return JsonResponse({})