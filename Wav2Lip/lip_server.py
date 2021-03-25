from lipsyn import LipSyner
from flask import Response
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import cv2
FRAME_BATCH_SIZE = 10
RESOURCE_PATH = '/home/ubuntu/resources/Wav2Lip/'

app = Flask(__name__)

model = LipSyner(RESOURCE_PATH + 'wav2lip_gan.pth', RESOURCE_PATH + '3min.mp4', RESOURCE_PATH + 'face_det_results_3min.pkl')

CORS(app)
@app.route('/lip', methods=['POST'])
def post_lip():
    audio_path   = request.json['audio_path']
    start_index  = request.json['start_index']
    def generate_frame():
        frame_batch = []
        check = False
        for frame in model.predict(audio_path,start_index):
            frame_batch.append(cv2.imencode('.jpg',frame)[1].tostring())
            if len(frame_batch) >= FRAME_BATCH_SIZE:
                
                if check == False:
                    yield b'endofframe'.join(frame_batch)
                    check = True
                else:
                    yield b'endofframe' + b'endofframe'.join(frame_batch)
                
                frame_batch = []
        
        if len(frame_batch) > 0:
            yield b'endofframe'.join(frame_batch)
    
    return Response(generate_frame(),content_type = 'application/octet-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5800,debug=False)
