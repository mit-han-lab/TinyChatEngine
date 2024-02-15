#!/bin/bash

# Clone whisper-c2translate
git clone https://github.com/Jiminator/whisper-ctranslate2.git
cd whisper-ctranslate2  
pip install -r requirements.txt
cd ../

# Clone EmotiVoice
git clone https://github.com/Jiminator/EmotiVoice.git 
pip install torch torchaudio 
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin pypinyin_dict   
cd EmotiVoice  
git clone https://www.modelscope.cn/syq163/WangZeJun.git  
git clone https://www.modelscope.cn/syq163/outputs.git
cd ../
