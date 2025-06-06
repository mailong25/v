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
from pydub import AudioSegment
import numpy as np
import librosa

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
setup_logger('clean_logger', 'clean.log', level=logging.INFO)
clean_logger = logging.getLogger('clean_logger')

BLENDER_URL = 'http://127.0.0.1:5901/predict'
LIPSYNC_URL = 'http://127.0.0.1:5800/lip'
SAMPLE_RATE = 44100
MAX_USERS = 5
INACTIVE_THRESHOLD = 30
ASR_SILENT_COUNT = -1
MESSAGE_BUSY = "Thanks for interesting in our service. The server is currently busy. Please try at another time."
MESSAGE_DUP = "Cleaning cache.... Please comeback after 1 minute"
MESSAGE_BACK  = {'spk':'bot','wav':'000','topic':'None','text': 'it is great to see you back , how is it going on ?'}
MESSAGE_START = {'spk':'bot','wav':'000','topic':'None','text': 'It is nice to meet you , how are you doing ?'}
MESSAGE_EMPTY = "i don't know"
TTS_AUDIO_PATH = '/home/ubuntu/chatbot/chatweb/files/audio/'
RESOURCE_PATH = '/home/ubuntu/resources/'

global_frames = global_var.video_frames.get_frames()

users = {}

with open('helping_words.txt') as f:
    helping_words = set(f.read().splitlines())

def random_id():
    return uuid.uuid4().hex

def user_on_leave(used_id):
    global users
    if used_id in users:
        cur_user = users[used_id]
        with open(cur_user['log_dialog'],'w') as f:
            json.dump(cur_user['history'],f)
        cur_user['asr'].stop()
        cur_user['logger'].handlers = []
        del cur_user['logger']
        del users[used_id]
        main_logger.info("Remove: " + used_id + " . Users: " + str(users.keys()))

def clean_space(args):
    global users
    while True:
        cur_time = datetime.now()
        clean_logger.info(str(cur_time))
        for user_id in list(users.keys()):
            inactive_seconds = (cur_time - users[user_id]['last_update']).total_seconds()
            clean_logger.info(" ".join([user_id, str(users[user_id]['last_update']),str(inactive_seconds)]))
            if inactive_seconds > INACTIVE_THRESHOLD:
                user_on_leave(user_id)
        
        temp_audios = [TTS_AUDIO_PATH + p for p in os.listdir(TTS_AUDIO_PATH)]
        for p in temp_audios:
            if cur_time.timestamp() - os.path.getctime(p) > 500:
                os.remove(p)
        
        time.sleep(5)

#Thread(target = clean_space, args = ("clean",)).start()

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
        main_logger.info("Reject user currently talking : " + user_id)
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
    cur_user['log'] = join_path(RESOURCE_PATH + 'logs',user_id)
    cur_user['log_dialog'] = join_path(cur_user['log'],'dialog.json')
    
    if os.path.exists(cur_user['log']):
        with open(cur_user['log_dialog']) as f:
            turns = json.load(f)
            if turns[-1]['spk'] == 'bot':
                turns[-1] = MESSAGE_BACK
            else:
                turns.append(MESSAGE_BACK)
        cur_user['history'] = turns
    else:
        os.mkdir(cur_user['log'])
        cur_user['history'] = [MESSAGE_START]

    with open(cur_user['log_dialog'],'w') as f:
        json.dump(cur_user['history'],f)
    
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

def video_feed(request):
    user_id = request.COOKIES['user_id']
    return StreamingHttpResponse(stream(user_id), content_type='multipart/x-mixed-replace; boundary=frame')

def blender_response(dialog_history, non_repeat):
    response = requests.post(BLENDER_URL, json = {"dialog": json.dumps(dialog_history), "non_repeat": non_repeat}).json()
    return {'spk':'bot','wav':'000','topic': response['topic'],'text': response['text']}

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

def sample_response(request):
    global users
    print("Generate sample !!")
    user_id = request.COOKIES['user_id']
    cur_user = users[user_id]
    sample = blender_response(cur_user['history'],False)['text']
    cur_user['asr'].final_text = []
    cur_user['asr'].final_result = []
    return JsonResponse({'text': sample})

def startfunc(request):
    global users
    user_id = request.COOKIES['user_id']
    cur_user = users[user_id]
    start = time.time()
    user_text = request.POST.get("text")
    if user_text != 'idle':
        response_json = blender_response(cur_user['history'],True)
        cur_user['history'].append(response_json)
        cur_user['last_update'] = datetime.now()
    
    response_text = cur_user['history'][-1]['text']
    
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
    global users
    cur_user = users[request.COOKIES['user_id']]
    ending_silent = AudioSegment.silent(1000,frame_rate = cur_user['sampling_rate'])._data
    cur_user['asr'].push_frames(ending_silent,cur_user['sampling_rate'])
    time.sleep(1.0)
    cur_user['asr'].is_done = True
    return JsonResponse({'text': ''})

def stop_asr(used_id):
    global users
    users[used_id]['asr'].stop()
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
    ajax_response = JsonResponse({'text': '', 'finish': 'false', 'score': ''})
    used_id = request.COOKIES['user_id']
    if used_id not in users:
        return ajax_response
    cur_user = users[used_id]
    data = request.FILES['audio']
    binaryHeader = data.read(44)
    frames = data.read()
    
    cur_user['last_update'] = datetime.now()
    
    try:
        if len(cur_user['asr'].recognizing_words) == 0:
            nchannels, sampwidth, framerate = extract_audio_info(binaryHeader)
            cur_user['writer'].setnchannels(nchannels)
            cur_user['writer'].setsampwidth(sampwidth)
            cur_user['writer'].setframerate(16000)
            cur_user['sampling_rate'] = framerate
    except:
        pass
    
    ## Convert sampling rate
    if cur_user['sampling_rate'] != 16000:
        array_frames = np.frombuffer(frames,dtype=np.int16)
        array_frames = array_frames.astype(np.float32, order='C') / 32768.0
        frames = librosa.resample(array_frames, cur_user['sampling_rate'], 16000, res_type='kaiser_best')
        frames = (frames * 32768.0).astype(np.int16,order='C').tobytes()
    
    if cur_user['asr'].is_inactive():
        user_on_leave(used_id)
        return JsonResponse({'text': "", 'finish': 'close', 'score': ''})
    
    if cur_user['asr'].state == "start":
        cur_user['asr'].push_frames(frames,16000)
        
        try:
            cur_user['writer'].writeframes(frames)
        except:
            pass
        
        recognize_text = cur_user['asr'].get_text()
        
        if cur_user['asr'].state == "start" and cur_user['asr'].is_done:
            
            #time.sleep(1)
            
            cur_user['asr'].state == "stop"
            print("stop reg !")
            
            Thread(target = stop_asr, args = (used_id,)).start()
            
#             if recognize_text == "":
#                 print("empty text, watting for 1 second")
            
            recognize_text = cur_user['asr'].get_text()
            recognize_result = cur_user['asr'].final_result
            
            fluency_score, pro_score = 100, 100
            
            if recognize_text != "":
                fluency_score = [res['PronunciationAssessment']['FluencyScore'] for res in recognize_result]
                fluency_score = int(sum(fluency_score) / len(fluency_score))
            
            if fluency_score < 85:
                fluency_string = " <font color=#FF0000>" + str(fluency_score) + "%" + "</font> "
            else:
                fluency_string = " <font color=#4169E1>" + str(fluency_score) + "%" + "</font> "
                
            word_scores = []
            pro_score = []
            for p in recognize_result:
                for word in p["Words"]:
                    word_lexicon = word['Word']
                    word_score = word['PronunciationAssessment']['AccuracyScore']
                    if word_lexicon in helping_words or "'" in word_lexicon:
                        word_score = 95
                    if word_score < 80:
                        word_lexicon = "<font color=#FF0000>" + word_lexicon + "</font>"
                    word_scores.append(word_lexicon)
                    pro_score.append(word_score)
            
            word_scores = " ".join(word_scores)
            
            if recognize_text != "":
                pro_score = int(sum(pro_score) / len(pro_score))
            else:
                pro_score = 100
            
            if pro_score < 85:
                pro_string = " <font color=#FF0000>" + str(pro_score) + "%" + "</font> "
            else:
                pro_string = " <font color=#4169E1>" + str(pro_score) + "%" + "</font> "
            
            score_string = "Fluency:" + fluency_string + "| Pronunciation:" + pro_string

            if recognize_text == "":
                recognize_text = MESSAGE_EMPTY
            
            cur_user['logger'].info("Response with: " + recognize_text)
            cur_user['history'].append({'spk':'user', 'wav':cur_user['user_audio'], 
                                        'topic':'None', 'text': recognize_text})
            
            word_scores += ' &nbsp; ==> Score:' + pro_string
            
            ajax_response = JsonResponse({'text': word_scores, 'finish': 'true', 'score': score_string})
    
    return ajax_response

def end(request):
    user_on_leave(request.COOKIES['user_id'])
    return JsonResponse({})

def clean(request):
    global users, clean_logger
    cur_time = datetime.now()
    clean_logger.info(str(cur_time))
    for user_id in list(users.keys()):
        inactive_seconds = (cur_time - users[user_id]['last_update']).total_seconds()
        clean_logger.info(" ".join([user_id, str(users[user_id]['last_update']),str(inactive_seconds)]))
        if inactive_seconds > INACTIVE_THRESHOLD:
            user_on_leave(user_id)

    temp_audios = [TTS_AUDIO_PATH + p for p in os.listdir(TTS_AUDIO_PATH)]
    for p in temp_audios:
        if cur_time.timestamp() - os.path.getctime(p) > 500:
            os.remove(p)
    return HttpResponse(MESSAGE_DUP)