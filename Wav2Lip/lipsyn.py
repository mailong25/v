from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
#import face_detection
from models import Wav2Lip
import platform
import time
import sys
import pickle

class Args:
    def __init__(self):
        self.checkpoint_path = ''
        self.face = ''
        self.audio = ''
        self.static = False
        self.pads = [0, 10, 0, 0]
        self.face_det_batch_size = 8
        self.wav2lip_batch_size = 32
        self.resize_factor = 1
        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.rotate = False
        self.nosmooth = False
        self.img_size = 96
        self.mel_step_size = 16
        self.fps = 24
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
args = Args()

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=args.device)

    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def datagen(frames, mels, face_results, batch_size = 128):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    face_det_results = face_results
    
    for i, m in enumerate(mels):
        
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        
        face = cv2.resize(face, (args.img_size, args.img_size))
        
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        
        if len(img_batch) >= batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

def _load(checkpoint_path):
    if args.device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(args.device)
    return model.eval()

class LipSyner:
    
    def __init__(self,checkpoint_path, video_path, face_det_results = None):
        args.face = video_path
        
        if not os.path.isfile(args.face):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(args.face)]
            args.fps = args.fps

        else:
            video_stream = cv2.VideoCapture(args.face)
            args.fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames...')

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if args.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

                if args.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)

        print ("Number of frames available for inference: "+str(len(full_frames)))

        if face_det_results == None:
            if args.box[0] == -1:
                if not args.static:
                    face_det_results = face_detect(full_frames.copy()) # BGR2RGB for CNN face detection
                else:
                    face_det_results = face_detect([full_frames.copy()[0]])
            else:
                y1, y2, x1, x2 = args.box
                face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        else:
            face_det_results = pickle.load(open(face_det_results,'rb'))
        
        self.model = load_model(checkpoint_path)
        self.full_frames = full_frames
        self.face_det_results = face_det_results
        self.num_frame = len(self.full_frames)
        print(len(list(self.predict('data/file.wav',0))))
        
    def predict(self,audio_path,start_index):
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        mel_chunks = []
        mel_idx_multiplier = 80./args.fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + args.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - args.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + args.mel_step_size])
            i += 1

        start_index = start_index % len(self.full_frames)
        select_frames = self.full_frames[start_index: start_index + len(mel_chunks)].copy()
        select_face_results = self.face_det_results[start_index: start_index + len(mel_chunks)].copy()

        if start_index + len(mel_chunks) > len(self.full_frames):
            loop_index = start_index + len(mel_chunks) - len(self.full_frames)
            select_frames += self.full_frames[:loop_index].copy()
            select_face_results += self.face_det_results[:loop_index].copy()

        gen = datagen(select_frames, mel_chunks, select_face_results, args.wav2lip_batch_size)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            
            batch_frames = []
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(args.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(args.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                yield f
