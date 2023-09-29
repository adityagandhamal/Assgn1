# Assignment No.1 - To create a Lip Sync Video

### Repo Structure

 - `checkpoints` - weights of the pre-trained model
   
 - `data_files` - input video and target audio to be synced
   
 - `face_detection` - a model to detect faces in a frame (ref. https://github.com/Rudrabha/Wav2Lip)
   
 - `models` - SOTA model for the lipsync task **Wav2Lip** (ref. https://github.com/Rudrabha/Wav2Lip)
   
 - `requirements.txt` - packages pinned to be installed
   
 - `sync.py` - the actual Python script to be run by the end-user
   
 - `utils.py` - a utility script for `sync.py`


## Steps to follow (Recommend using a UNIX with a conda environment)

1. Create a virual environment `conda create --name listed_1 python=3.6`
   
2. Clone the repo `git clone https://github.com/adityagandhamal/Assgn1.git`
   
3. run `cd Assgn1`
   
4. run `pip install -r requirements.txt` [_Note:_ _Go on installing each package if the process gets stuck (happens usually while building dependency wheels)_]
   
5. Download the face detection [model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) and place it in `face_detection/detection/sfd/` as `s3fd.pth`
    
6. Download the weights of the pre-trained model [Wav2Lip + GAN](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) and place the file in `checkpoints`
    
7. Run `python sync.py`
    
8. You'll obtain an output `listed_out.mp4`

## Sample Input and Output

### Input Video
https://github.com/adityagandhamal/Assgn1/assets/61016383/4e130cfb-0ca0-4b1a-a213-49f23665d60b

### Target Audio
https://github.com/adityagandhamal/Assgn1/assets/61016383/fbfc890b-95b5-4be4-b088-552498ae2442

### Output Result
https://github.com/adityagandhamal/Assgn1/assets/61016383/03f26811-9ecc-46b4-b8e4-24f7c0f6064c

# Disclaimer:

The sample above is just a demo to get a notion of the task while the actual output video has been attached as a drive link in the mail. Keeping in mind the limitations of the pre-trained model and the scope of the input video (as the subject is seen to be disappearing from the scene frequently), the video and the audio are both trimmed using a third-party website. 

Also, make sure to run the code on a GPU instance as the process is **killed** on a CPU. The following attached is the proof of the same.

![Screenshot (1392)](https://github.com/adityagandhamal/Assgn1/assets/61016383/47f01bd7-ce8a-480b-a811-ebb9910541eb)


#### Hence, the code is run in a [Colab Notebook](https://colab.research.google.com/drive/17v70lBAieKJFh_1ShkyKI2pcJ_dTeVBl?usp=sharing)



