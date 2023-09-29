from models import Wav2Lip
import torch
import numpy as np
import face_detection
import tqdm
import cv2

# CONSTANTS
_BATCH_SIZE = 16
_MEL_STEP_SIZE = 16
_IMG_SIZE = 96
_WAV2LIP_BATCH_SIZE = 128

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
	
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, 
                                            device=device)
    
    batch_size = _BATCH_SIZE
    
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
    pady1, pady2, padx1, padx2 = [0, 0, 0, 0]
    for rect, image in zip(predictions, images):
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])
        
    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    
    del detector
    return results

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
    
    for i, m in enumerate(mels):
        idx = i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        
        face = cv2.resize(face, (_IMG_SIZE, _IMG_SIZE))
        
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        
        if len(img_batch) >= _WAV2LIP_BATCH_SIZE:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            
            img_masked = img_batch.copy()
            img_masked[:, _IMG_SIZE//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
            
            
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        
        img_masked = img_batch.copy()
        img_masked[:, _IMG_SIZE//2:] = 0
        
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        
        yield img_batch, mel_batch, frame_batch, coords_batch
        
mel_step_size = _MEL_STEP_SIZE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print('Using {} for inference.'.format(device))
            
def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, 
                                loc: storage)
    return checkpoint


def load_model(path):
	model = Wav2Lip()
	#print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()
        