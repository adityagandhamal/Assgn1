import cv2
import torch 
import tqdm
import numpy as np
import librosa 
import subprocess, platform
from utils import device, datagen, load_model, _MEL_STEP_SIZE, _WAV2LIP_BATCH_SIZE

#args = args.face, args.audio, args.outfile, args.checkpoint device

_RESIZE_FACTOR = 2

print("device: ", device)

def main(vid, audio, checkpoint, device, outfile):
    
    video_stream = cv2.VideoCapture(vid)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    print('Reading video frames...')

    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        
        
        frame = cv2.resize(frame, (frame.shape[1]//_RESIZE_FACTOR, frame.shape[0]//_RESIZE_FACTOR))
            
        y1, y2, x1, x2 = [0, -1, 0, -1]
        
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        
        frame = frame[y1:y2, x1:x2]
        
        full_frames.append(frame)
        
    print ("Number of frames available for inference: "+str(len(full_frames)))

    wav, sr = librosa.load(audio, sr=16000)
    #mel = audio.melspectrogram(wav)
    
    mel = librosa.feature.melspectrogram(y=wav, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    print(mel.shape)

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + _MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - _MEL_STEP_SIZE:])
            
        mel_chunks.append(mel[:, start_idx : start_idx + _MEL_STEP_SIZE])
        i += 1
        
    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = _WAV2LIP_BATCH_SIZE
    gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            model = load_model(checkpoint) # path to checkpoint
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', 
                                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

        out.release()
    
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio, 'temp/result.avi', outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')


video_path = "data_files/vid_trim.mp4"
audio_path = "data_files/audio_trim.wav"
chk = "checkpoints/wav2lip_gan.pth"
device = device
out = "sol_out.mp4"

main(video_path, audio_path, chk, device, out)
    