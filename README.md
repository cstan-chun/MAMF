#  MAMF: Mutual Attention-Based Multimodal Fusion for Audio-Visual Emotion Recognition
## Environment setup
1. create a new environment using conda or pip (python==3.8)
2. pip install -r requirements.txt
## Download Data
The two datasets (RAVDESS,IEMOCAP) are available from this link:
1. [https://zenodo.org/records/1188976#.ZECaH3ZBw2x](https://zenodo.org/records/1188976#.ZECaH3ZBw2x)  This data is divided into song and speech, and you only need to download the speech.
2. [https://gitcode.com/Open-source-documentation-tutorial/0e833/tree/main](https://gitcode.com/Open-source-documentation-tutorial/0e833/tree/main)
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
cd ravdess_preprocessing  
python extractor_iemocap.py
## train

