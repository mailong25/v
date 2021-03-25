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
import pickle

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    if len(l.handlers) == 0:
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        fileHandler = logging.FileHandler(log_file, mode='a')
        fileHandler.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fileHandler)

setup_logger('main_logger', 'log.log', level=logging.INFO)
main_logger = logging.getLogger('main_logger')

BLENDER_URL = 'http://54.179.106.170:5901/predict'
LIPSYNC_URL = 'http://54.179.106.170:5800/lip'
SAMPLE_RATE = 44100
MAX_USERS = 5
INACTIVE_THRESHOLD = 20
ASR_SILENT_COUNT = 4
MESSAGE_BUSY = "Thanks for interesting in our service. The server is currently busy. Please try at another time."
MESSAGE_DUP = "You are currently talking to V."
MESSAGE_BACK = 'bot\t000\tIt is great to see you back. What do you wanna talk about today ?'
MESSAGE_START = 'bot\t000\tIt is nice to meet you. I am V. How are you doing ?'
TTS_AUDIO_PATH = '/home/ubuntu/chatbot/chatweb/files/audio/'

global_frames = global_var.video_frames.get_frames()

users = {}

def random_id():
    return uuid.uuid4().hex

def user_on_leave(used_id):
    global users
    if used_id in users:
        cur_user = users[used_id]
        with open(cur_user['log_dialog'],'w') as f:
            f.write('\n'.join(cur_user['history']))
        cur_user['asr'].stop()
        cur_user['logger'].handlers = []
        del cur_user['logger']
        del users[used_id]
        main_logger.info("Remove: " + used_id + " . Users: " + str(users.keys()))

def clean_space(args):
    global users
    while True:
        cur_time = datetime.now()
        for user_id in list(users.keys()):
            inactive_seconds = (cur_time - users[user_id]['last_update']).total_seconds()
            if inactive_seconds > INACTIVE_THRESHOLD:
                user_on_leave(user_id)
        
        temp_audios = [TTS_AUDIO_PATH + p for p in os.listdir(TTS_AUDIO_PATH)]
        for p in temp_audios:
            if cur_time.timestamp() - os.path.getctime(p) > 500:
                os.remove(p)
        
        time.sleep(5)

Thread(target = clean_space, args = ("clean",)).start()

# Create your views here.
def index(request):
    global users
    if len(users) > MAX_USERS:
        return HttpResponse(MESSAGE_BUSY)
    
    template = loader.get_template('index.html')
    response = HttpResponse(template.render({}, request))
    user_id = request.COOKIES.get("user_id",None)
    
    if user_id == None:
        user_id = uuid.uuid4().hex
        response.set_cookie("user_id",user_id,10000000,datetime.now() + timedelta(days=3650))
        main_logger.info("New user_id : " + user_id)
    else:
        main_logger.info("Existing user_id : " + user_id)
    
    try:
        main_logger.info("User agent : " + str(request.META['HTTP_USER_AGENT']))
    except:
        pass
    
    if user_id in users:
        return HttpResponse(MESSAGE_DUP)
    else:
        return response

def new(request):
    global users
    user_id = request.COOKIES.get("user_id")
    main_logger.info("Init: " + user_id)
    users[user_id] = {'last_update': datetime.now(),'cur_frame_id': 0,
                      'lip': {}, 'writer': None, 'tts': tts.TTS(), 'sampling_rate': SAMPLE_RATE,
                      'asr' : asr.ASR(auto_stop_duration = ASR_SILENT_COUNT, inactive_mins = 5)}

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
    
    setup_logger(user_id,join_path(cur_user['log'],'log.log'), level=logging.INFO)
    cur_user['logger'] = logging.getLogger(user_id)
    cur_user['logger'].info("New session started !")
    
    main_logger.info(str(users.keys()))
    return JsonResponse({})

def stream(user_id):
    global global_frames, users
    users[user_id]['logger'].info("entering stream!")
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
    user_id = request.COOKIES['user_id']
    return StreamingHttpResponse(stream(user_id), content_type='multipart/x-mixed-replace; boundary=frame')

def blender_response(dialog_history):
    dialog_history = [t.split('\t')[2] for t in dialog_history]
    context = '\n'.join(dialog_history)
    response = requests.post(BLENDER_URL, json = {"context": context})
    return 'bot\t000\t' + response.json()['response']['text']

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
        cur_user['history'].append('user\t' + cur_user['user_audio'] + '\t' + user_text)
        response_text = blender_response(cur_user['history'])
        cur_user['history'].append(response_text)
        cur_user['last_update'] = datetime.now()
    
    response_text = response_text.split('\t')[2]
    cur_user['logger'].info("Blender for " + response_text + " is: " + str(time.time() - start))
    
    response_audio = TTS_AUDIO_PATH + uuid.uuid4().hex + ".mp3"
    response_audio_wav = response_audio.replace('.mp3','.wav')
    response_audio_mp3 = response_audio[:-5] + '.mp3'
    
    cur_user['tts'].predict(response_text,response_audio)
    
    convert_cmd = 'ffmpeg -i ' + response_audio + ' ' + response_audio_wav
    Popen(convert_cmd.split(), stdout=PIPE, stderr=PIPE).wait()
    
    convert_cmd = 'ffmpeg -i ' + response_audio + ' -ar 44100 ' + response_audio_mp3
    Popen(convert_cmd.split(), stdout=PIPE, stderr=PIPE).wait()
    
    audio_len = int(MP3(response_audio).info.length * 1000)
    audio_url = "https://www.vvreborn.com/files/audio/" + ntpath.basename(response_audio_mp3)
    
    cur_user['logger'].info("TTS: " + str(time.time() - start))
    
    del cur_user['lip']
    cur_user['lip'] = {}
    
    start_index = cur_user['cur_frame_id'] + 20
    
    thread = Thread(target = updating_frame, args = (response_audio_wav, start_index, user_id))
    thread.start()
    
    while(len(cur_user['lip']) == 0):
        time.sleep(0.1)
    cur_user['logger'].info("Pre-Lip: " + str(time.time() - start))
    
    if start_index < cur_user['cur_frame_id']:
        cur_user['cur_frame_id'] = start_index
    
    SLEEP_TIME = float(start_index - cur_user['cur_frame_id']) / 24.0
    SLEEP_TIME = max(0,SLEEP_TIME - 0.5)
    time.sleep(SLEEP_TIME)
    cur_user['logger'].info("Lip: " + str(time.time() - start))
    return JsonResponse({'text': response_text,'audio':audio_url, 'audio_len' : audio_len})

def start_reg(request):
    global users
    cur_user = users[request.COOKIES['user_id']]
    if cur_user['asr'].state != "start":
        cur_user['asr'].start()
        cur_user['user_audio'] = random_id()
        cur_user['writer'] = wave.open(cur_user['log'] + '/' + cur_user['user_audio'] + '.wav', 'wb')
    return JsonResponse({})

def stop_reg(request):
    return None

def stop_asr(used_id):
    global users
    users[used_id]['asr'].stop()
    time.sleep(3)
    users[used_id]['writer'].close()

def extract_audio_info(binary_data):
    temp_path = random_id()
    with open(temp_path,'wb') as f:
        f.write(binary_data)
    info = wave.open(temp_path, 'rb')
    os.remove(temp_path)
    return info.getnchannels(), info.getsampwidth(), info.getframerate()

def recognize(request):
    global users, file_count
    ajax_response = JsonResponse({'text': '', 'finish': 'false'})
    used_id = request.COOKIES['user_id']  
    if used_id not in users:
        return ajax_response
    cur_user = users[used_id]
    data = request.FILES['audio']
    binaryHeader = data.read(44)
    frames = data.read()
    
    try:
        if len(cur_user['asr'].recognizing_words) == 0:
            nchannels, sampwidth, framerate = extract_audio_info(binaryHeader)
            cur_user['writer'].setnchannels(nchannels)
            cur_user['writer'].setsampwidth(sampwidth)
            cur_user['writer'].setframerate(framerate)
            cur_user['sampling_rate'] = framerate
    except:
        pass
    
    if cur_user['asr'].is_inactive():
        user_on_leave(used_id)
        return JsonResponse({'text': "", 'finish': 'close'})
    
    if cur_user['asr'].state == "start":
        cur_user['asr'].push_frames(frames,cur_user['sampling_rate'])
        cur_user['writer'].writeframes(frames)
        recognize_text = cur_user['asr'].get_text()
        
        if cur_user['asr'].state == "start" and cur_user['asr'].is_done():
            print("stop reg !")
            Thread(target = stop_asr, args = (used_id,)).start()
            cur_user['logger'].info("Response with: " + recognize_text)
            ajax_response = JsonResponse({'text': recognize_text, 'finish': 'true'})
    
    return ajax_response

def end(request):
    user_on_leave(request.COOKIES['user_id'])
    return JsonResponse({})
