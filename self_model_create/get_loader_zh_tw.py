import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import jieba


class Vocabulary:
    def __init__(self, freq_threshold):
        """
        初始化 Vocabulary 類別。

        Parameters:
        - freq_threshold (int): 用於構建詞彙表時的頻率閾值。
        """
        # 設定標準值
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        """
        取得詞彙表的長度。

        Returns:
        - int: 詞彙表的總數。
        """
        return len(self.itos)

    def numericalize(self, sentence):
        """
        將文本數值化，將單詞轉換為對應的索引。

        Parameters:
        - sentence (str): 要數值化的文本。

        Returns:
        - list: 數值化後的索引列表。
        """
        return [self.stoi[word] if word in self.stoi else self.stoi["<UNK>"] for word in self.tokenizer_zh(sentence)]
    
    @staticmethod
    def tokenizer_zh(text):
        """
        將中文文本分詞成單詞。

        Parameters:
        - text (str): 要分詞的中文文本。

        Returns:
        - list: 分詞後的單詞列表。
        """
        return [tok for tok in jieba.cut(text, cut_all=True)]

    def build_vocabulary(self, sentence_list):
        """
        建立詞彙表，將單詞添加到 itos 和 stoi 中。

        Parameters:
        - sentence_list (list): 包含多個句子的列表。
        """
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_zh(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

class CoCoDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=1):
        """
        初始化 CoCoDataset 類別。

        Parameters:
        - root_dir (str): 包含圖像的根目錄的路徑。
        - captions_file (str): 包含圖像標註資訊的 CSV 檔案的路徑。
        - transform (callable, optional): 圖像轉換的函數。預設為 None。
        - freq_threshold (int, optional): 建立詞彙表時的頻率閾值。預設為 5。
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # 取得 img 和 caption 欄位
        self.imgs = self.df["image"]
        self.captions = self.df["text"]

        # 初始化詞彙表並建立詞彙表
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        """
        取得資料集的長度。

        Returns:
        - int: 資料集的總數。
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        取得指定索引的圖像和標註。

        Parameters:
        - index (int): 資料集中的索引。

        Returns:
        - tuple: 包含圖像和數值化標註的元組。
        """
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        """
        初始化 MyCollate 類別。

        Parameters:
        - pad_idx (int): 用於填充序列的索引。
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        將一個批次的樣本整理成模型所需的格式。

        Parameters:
        - batch (list): 一個批次的樣本，每個樣本是一個元組(img, target)。

        Returns:
        - tuple: 包含整理後的圖像和標註的元組。
        """
        # 提取圖像並將它們串聯在一起
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        # 提取標註並進行填充
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

from torch.utils.data import DataLoader

def get_loader_zh_tw(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    """
    創建並返回一個 DataLoader 對象和相應的資料集。

    Parameters:
    - root_folder (str): 包含圖像的根目錄的路徑。
    - annotation_file (str): 包含圖像標註資訊的 CSV 檔案的路徑。
    - transform (callable): 圖像轉換的函數。
    - batch_size (int, optional): 每個批次的樣本數。預設為 32。
    - num_workers (int, optional): 資料載入時使用的子進程數。預設為 8。
    - shuffle (bool, optional): 是否對資料進行洗牌。預設為 True。
    - pin_memory (bool, optional): 是否將數據複製到 CUDA 固定記憶體。預設為 True。

    Returns:
    - tuple: 包含 DataLoader 對象和對應的資料集。
    """
    # 創建 CoCoDataset 對象
    dataset = CoCoDataset(root_folder, annotation_file, transform=transform)

    # 取得填充索引
    pad_idx = dataset.vocab.stoi["<PAD>"]

    # 創建 DataLoader 對象，並使用自定義的 MyCollate 作為 collate_fn
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset
