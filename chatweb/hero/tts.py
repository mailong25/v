from contextlib import closing
import boto3

AWS_ACCESS_KEY_ID="AKIA2PAKAC3NN756LZ76"
AWS_SECRET_ACCESS_KEY="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"
REGION_NAME="ap-southeast-1"

class TTS:
    
    def __init__(self,format_ = 'mp3'):
        self.polly_client = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID,
                                     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                     region_name=REGION_NAME).client('polly')
        self.format = format_
        
    def predict(self,text,save_path):
        response = self.polly_client.synthesize_speech(Engine='neural', TextType='ssml',
#             Text='<speak><prosody rate="100%">' + text + '</prosody> <break time="1s"/> </speak>',
              Text='<speak>' + text + '</speak>',
            OutputFormat=self.format,VoiceId="Salli",SampleRate='16000')

        with closing(response["AudioStream"]) as stream:
            with open(save_path, "wb") as file:
                file.write(stream.read())
                file.close()