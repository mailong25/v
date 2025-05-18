# from contextlib import closing
# import boto3

# AWS_ACCESS_KEY_ID=""
# AWS_SECRET_ACCESS_KEY=""
# REGION_NAME=""

# class TTS:
    
#     def __init__(self,format_ = 'mp3'):
#         self.polly_client = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID,
#                                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#                                      region_name=REGION_NAME).client('polly')
#         self.format = format_
        
#     def predict(self,text,save_path):
#         response = self.polly_client.synthesize_speech(Engine='neural', TextType='ssml',
#               Text='<speak>' + text + '</speak>',
#             OutputFormat=self.format,VoiceId="Salli",SampleRate='16000')

#         with closing(response["AudioStream"]) as stream:
#             with open(save_path, "wb") as file:
#                 file.write(stream.read())
#                 file.close()

from transformers import VitsModel, AutoTokenizer
import torch
import scipy

class TTS:
    
    def __init__(self):
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    def predict(self, text, save_path):
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model(**inputs).waveform.float().numpy()
        
        scipy.io.wavfile.write(save_path, rate=model.config.sampling_rate, data=output)
        
