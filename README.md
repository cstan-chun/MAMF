#  MAMF: Mutual Attention-Based Multimodal Fusion for Audio-Visual Emotion Recognition
## Environment setup
1. create a new environment using conda or pip (python==3.8)
2. pip install -r requirements.txt
## Download Data
The two datasets (RAVDESS,IEMOCAP) are available from this link.About RAVDESS,this data is divided into song and speech, and you only need to download the speech:  
1.RAVDESS: [https://zenodo.org/records/1188976#.ZECaH3ZBw2x](https://zenodo.org/records/1188976#.ZECaH3ZBw2x)  
2.IEMOCAP: [https://gitcode.com/Open-source-documentation-tutorial/0e833/tree/main](https://gitcode.com/Open-source-documentation-tutorial/0e833/tree/main)
## data processing
### RAVDESS
Preprocessing scripts are located in ravdess_preprocessing/ Inside each of three scripts, specify the path (full path!) where you have downloaded the data. Then run:  
```
cd ravdess_preprocessing  
python extract_faces.py  
python extract_audios.py  
python create_annotations.py
```
### IEMOCAP
```
cd ravdess_preprocessing  
python extractor_iemocap.py
```
As a result you will have annotations.txt file that you can use further for training.
## training
For visual module weight initialization, download the pre-trained EfficientFace from here under 'Pre-trained models'. In our experiments, we use the model pre-trained on AffectNet7, i.e., EfficientFace_Trained_on_AffectNet7.pth.tar. If you want to use a different one, download it and later specify the path in --pretrain_path argument to main.py. Otherwise, you can ignore this step and train from scratch (although you will likely obtain lower performance).  

### For training,You can check or change some parameters in the opts.py, then run:
```
python main.py
```

