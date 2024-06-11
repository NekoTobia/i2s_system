import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        """
        初始化 EncoderCNN 類別。

        Parameters:
        - embed_size (int): 嵌入向量的大小。
        - train_CNN (bool, optional): 決定是否訓練 CNN 模型的標誌。預設為 False。
        """
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN

        # 載入預訓練的 vit_b_16 模型，並替換最後一層全連接層
        self.vit = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, embed_size)

        # 定義 ReLU 和 Dropout 層
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        """
        定義前向傳播方法。

        Parameters:
        - images (torch.Tensor): 輸入的圖像張量。

        Returns:
        - torch.Tensor: 經過模型前向傳播後的特徵張量。
        """
        # 通過 vit_b_16 獲取圖像特徵
        features = self.vit(images)

        # 選擇性地設定要訓練的參數
        for name, param in self.vit.named_parameters():
            if "heads.weight" in name or "heads.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        # 使用 ReLU 和 Dropout 進行後處理
        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """
        初始化 DecoderRNN 類別。

        Parameters:
        - embed_size (int): 嵌入向量的大小。
        - hidden_size (int): LSTM 隱藏層的大小。
        - vocab_size (int): 詞彙表的大小。
        - num_layers (int): LSTM 的層數。
        """
        super(DecoderRNN, self).__init__()

        # 定義嵌入層、LSTM 層、全連接層和 Dropout 層
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        定義前向傳播方法。

        Parameters:
        - features (torch.Tensor): 圖像特徵的張量。
        - captions (torch.Tensor): 解碼器的輸入序列。

        Returns:
        - torch.Tensor: 經過模型前向傳播後的輸出序列。
        """
        # 嵌入詞彙索引，並進行 Dropout
        embeddings = self.dropout(self.embed(captions))

        # 將圖像特徵和嵌入詞彙序列串聯在一起
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        # 將串聯後的序列通過 LSTM 層
        hiddens, _ = self.lstm(embeddings)

        # 通過全連接層獲取輸出序列
        outputs = self.linear(hiddens)

        return outputs

import torch.nn as nn

class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers,vocoder='WaveRNN'):
        """
        初始化 CNNtoLSTM 類別。

        Parameters:
        - embed_size (int): 嵌入向量的大小。
        - hidden_size (int): LSTM 隱藏層的大小。
        - vocab_size (int): 詞彙表的大小。
        - num_layers (int): LSTM 的層數。
        """
        super(CNNtoLSTM, self).__init__()

        # 創建 EncoderCNN 和 DecoderRNN 實例
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, images, captions):
        """
        定義前向傳播方法。

        Parameters:
        - images (torch.Tensor): 輸入的圖像張量。
        - captions (torch.Tensor): 解碼器的輸入序列。

        Returns:
        - torch.Tensor: 經過模型前向傳播後的輸出序列。
        """
        # 通過 EncoderCNN 獲取圖像特徵
        features = self.encoderCNN(images)

        # 通過 DecoderRNN 進行解碼
        outputs = self.decoderRNN(features, captions)

        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        
        """
        給定一張圖像，生成對應的標註。

        Parameters:
        - image (torch.Tensor): 輸入的圖像張量。
        - vocabulary (Vocabulary): 詞彙表的實例。
        - max_length (int, optional): 生成標註的最大長度。預設為 50。

        Returns:
        - list: 生成的標註序列。
        """
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return_data = [vocabulary.itos[idx] for idx in result_caption][1:-2]
        return_data = " ".join(return_data)
        
        return return_data
