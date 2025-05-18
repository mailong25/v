import wave
import time
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import PronunciationAssessmentConfig as ProConfig
from azure.cognitiveservices.speech import PronunciationAssessmentGradingSystem as ProGrade
from azure.cognitiveservices.speech import PronunciationAssessmentGranularity as ProGran
import numpy as np
import librosa
import requests
import json

def get_vad(frame_bytes):
    res = requests.post('http://127.0.0.1:5700/predict', data = frame_bytes)
    res = res.json()['result']
    return res

class ASR:
    
    def __init__(self,key = "", 
                 region = "", auto_stop_duration = 3, inactive_mins = 5):
        
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.speech_recognition_language="en-US"
        #speech_config.enable_dictation()
        speech_config.set_profanity(speechsdk.ProfanityOption.Raw)
        self.pro_config = ProConfig(reference_text=None, grading_system=ProGrade.HundredMark,granularity=ProGran.Word)
        self.auto_stop_duration = auto_stop_duration
        self.speech_config = speech_config
        self.final_text = []
        self.final_result = []
        self.recognizing_words = []
        self.recognizing_states = []
        self.minimum_state = int(self.auto_stop_duration)
        self.state = "stop"
        self.inactive_threshold = inactive_mins * 60
        self.is_done = False
        self.inactive_count = 0
        
    def start(self):
        if self.state == "stop":
            def get_result(evt):
                nonlocal self
                if evt.result.text == '':
                    return
                self.recognizing_words += [True,False]
                self.final_text.append(evt.result.text)
                self.final_result.append(json.loads(evt.result.json)['NBest'][0])
            def notify_recognizing(evt):
                nonlocal self
                if len(evt.result.text) >= 1:
                    self.recognizing_words.append(True)

            self.stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=self.stream)
            self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            self.pro_config.apply_to(self.speech_recognizer)
            self.speech_recognizer.recognized.connect(get_result)
            self.speech_recognizer.recognizing.connect(notify_recognizing)
            self.speech_recognizer.start_continuous_recognition_async()
            self.final_text = []
            self.final_result = []
            self.recognizing_words = []
            self.recognizing_states = []
            self.state = "start"
            self.is_done = False
            self.inactive_count = 0
    
    def push_frames(self, frames, sampling_rate):
        
        self.inactive_count += 1
        
        if self.state == "start":
            if len(self.recognizing_words) > 0:
                self.recognizing_states.append(self.recognizing_words[-1])
                self.recognizing_words.append(self.recognizing_words[-1])
            else:
                self.recognizing_states.append(False)

            if sampling_rate != 16000:
                array_frames = np.frombuffer(frames,dtype=np.int16)
                array_frames = array_frames.astype(np.float32, order='C') / 32768.0
                frames = librosa.resample(array_frames, sampling_rate , 16000, res_type='kaiser_best')
                frames = (frames * 32768.0).astype(np.int16,order='C').tobytes()
            
            try:
                self.stream.write(frames)
            except:
                print("stream error! ")
            
            if len(self.recognizing_words) >= 8 and self.recognizing_words[-1] == False:
                if True not in self.recognizing_words[-8:]:
                    self.is_done = True
            
            if self.auto_stop_duration > 0:
                is_speech = True
                try:
                    is_speech = len(get_vad(frames)) > 0
                except:
                    print("VAD error: ")
            
                self.recognizing_states.append(is_speech)

                if len(self.recognizing_words) >= self.minimum_state and self.recognizing_words[-1] == False:
                    if len(self.recognizing_states) >= self.minimum_state:
                        if self.recognizing_states[-self.minimum_state:].count(True) <= 0:
                            self.is_done = True

    def is_inactive(self):
        if self.inactive_count > self.inactive_threshold:
            return True
        return False
    
    def get_text(self):
        return ' '.join(self.final_text)
    
    def stop(self):
#         if self.state == "start":
        self.state = "stop"
#             self.final_text = []
        self.recognizing_words = []
        self.recognizing_states = []
        try:
            self.stream.close()
            self.speech_recognizer.stop_continuous_recognition_async()
            del self.stream, self.speech_recognizer
        except:
            pass
