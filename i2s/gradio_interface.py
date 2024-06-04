#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import json
import warnings
warnings.filterwarnings('ignore')
#!pip install open_clip_torch
import torch
#from zoedepth.utils.misc import get_image_from_url, colorize
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS 
import os
from gtts import gTTS
from pygame import mixer
import tempfile
import cv2
import numpy as np
import time
from flask import Flask, request, render_template, send_file
import os
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import gradio as gr
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from get_loader import get_loader
from get_loader_zh import get_loader_zh
from model import CNNtoLSTM
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

device = "cuda:0" if torch.cuda.is_available() else "cpu"
audio_filename = 'audio.mp3'


model_name_blip2 = 'Salesforce/blip2-opt-2.7b'
processor_blip2 = Blip2Processor.from_pretrained(model_name_blip2)
model_blip2 = Blip2ForConditionalGeneration.from_pretrained(
    model_name_blip2, device_map={"": 0}, load_in_4bit=True
)



def image_captioning_blip2vqa(image,question):
    image = Image.open(image)
    print(device)
    inputs = processor_blip2(images=image,text=question,return_tensors="pt").to(device)  # prompt
    generated_ids = model_blip2.generate(**inputs)
    result = processor_blip2.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result
    
def image_captioning_blip2(image):
    print(device)
    image = Image.open(image)
    inputs = processor_blip2(images=image,return_tensors="pt").to(device)  # prompt
    generated_ids = model_blip2.generate(**inputs)
    result = processor_blip2.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result



#=====================================================
#self model
model = torch.load('./self_model/finish_model_RAdam_0.01.pkl')
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )
train_loader, dataset = get_loader(
    root_folder="./",
    annotation_file="mscoco_data.csv",
    transform=transform,
    num_workers=8,
    batch_size = 128,
)


model_zh=torch.load('./self_model/finish_model_NAdam_1.pkl')
test_loader_zh, dataset_zh = get_loader_zh(
    root_folder="./",
    annotation_file="zh_data_mscoco_vizwiz.csv",
    transform=transform,
    num_workers=8,
    batch_size = 128,
)


#=====================================================
#self model

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def self_model(image):
    image = Image.open(image)
    test_img = transform(image).unsqueeze(0).to(device)
    text = model.caption_image(test_img, dataset.vocab)  
    return text
def self_model_zh(image):
    image = Image.open(image)
    test_img = transform(image).unsqueeze(0).to(device)
    text = model_zh.caption_image(test_img, dataset_zh.vocab)  
    return text



#=====================================================
#tts
def tts_en(sentence):
    tts = gTTS(text=sentence, lang='en')
    tts.save(audio_filename)
def tts_zh(sentence):
    tts = gTTS(text=sentence, lang='zh-tw')
    tts.save(audio_filename)

    
#=====================================================
#導向
def v2v(image, selected_model,question="text"):
    if selected_model == "BLIP-2 VQA":
        sentence = image_captioning_blip2vqa(image,question)
        tts_en(sentence)
    elif selected_model == "BLIP-2":
        sentence = image_captioning_blip2(image)
        tts_en(sentence)
    elif selected_model=='Our Model zh-tw':
        sentence=self_model_zh(image)
        tts_zh(sentence)
    else:
        sentence = self_model(image)
        tts_en(sentence)

    return audio_filename, sentence


# Gradio interface with model selection dropdown
app = gr.Interface(
    fn=v2v,
    inputs=[gr.Image(type = 'filepath'),gr.Radio(["BLIP-2",'BLIP-2 VQA', "Our Model en","Our Model zh-tw"]),"text"],  # Radio buttons for model choice
    outputs=["audio", "text"]
)
app.launch(share = True)

