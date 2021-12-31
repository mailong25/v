from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
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
        self.face_det_batch_size = 8
        self.wav2lip_batch_size = 32
        self.img_size = 96
        self.mel_step_size = 16
        self.fps = 25
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
args = Args()

def _load(checkpoint_path):
    if args.device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,map_location='cpu')
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
    model.half()
    model2 = model.to(args.device)
    del model
    return model2.eval()

class LipSyner:
    def __init__(self,checkpoint_path, video_path, face_path):
        self.video_capture = cv2.VideoCapture(video_path)
        self.frame_w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video = video_path
        self.cache_seconds = 20
        
        self.cache_frames = []
        while True:
            still_reading, frame = self.video_capture.read()
            self.cache_frames.append(frame)
            if len(self.cache_frames) > self.cache_seconds * self.fps:
                break
        
        self.model = load_model(checkpoint_path)
        self.face_imgs, self.face_coords = pickle.load(open(face_path,'rb'))
        for _ in self.predict('data/file.wav',0):
            pass
        
    def get_frame_by_index(self,start_idx, end_idx):
        video_capture = cv2.VideoCapture(self.video)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        select_frames = self.cache_frames.copy()
        
        total_read_frame = 0
        while total_read_frame <= end_idx - start_idx:
            still_reading, frame = video_capture.read()
            if not still_reading:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES,0)
            else:
                select_frames[total_read_frame] = frame
                total_read_frame += 1

        return select_frames[:total_read_frame]
        
    def get_face_by_index(self,start_idx, end_idx):
        select_face_imgs   = self.face_imgs[start_idx:end_idx].copy()
        select_face_coords = self.face_coords[start_idx:end_idx].copy()

        if end_idx > self.num_frames:
            loop_index = end_idx - self.num_frames
            select_face_imgs = np.vstack((select_face_imgs, self.face_imgs[:loop_index].copy()))
            select_face_coords += self.face_coords[:loop_index].copy()
        
        return select_face_imgs, select_face_coords

    def datagen(self, start_index, mels, face_imgs, face_coords, batch_size = 32):    
        for i in range(0,len(mels),batch_size):
            batch_end = min(i + batch_size, len(mels))
            img_batch = face_imgs[i:batch_end].copy()
            mel_batch = mels[i:batch_end].copy()
            frame_batch = self.get_frame_by_index(start_index + i, start_index + batch_end)
            coords_batch = face_coords[i:batch_end].copy()
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch

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
        mel_chunks = np.asarray(mel_chunks)
        
        start_idx = start_index % self.num_frames
        end_idx   = start_index + len(mel_chunks)
        
        #select_frames = self.get_frame_by_index(start_idx, end_idx)
        select_face_imgs, select_face_coords  = self.get_face_by_index(start_idx, end_idx)
        gen = self.datagen(start_idx, mel_chunks, select_face_imgs, select_face_coords, args.wav2lip_batch_size)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).half().to(args.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).half().to(args.device)
            
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
           
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                yield f
