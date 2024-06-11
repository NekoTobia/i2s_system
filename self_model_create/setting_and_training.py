import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml
import pandas as pd
import os 


from get_loader import get_loader
from model import CNNtoLSTM
from EarlyStopping import EarlyStopping
import transform

class ViTLSTM:
    def __init__(self):
        with open('train_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        self.num_workers = config['num_workers']
        self.batch_size = config['batch_size']
        self.train_CNN = config['train_CNN']
        self.embed_size = config['embed_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']
        self.gamma = config['gamma']
        self.scheduler_step_size = config['scheduler_step_size']
        self.opti = config['opti']
        
    def get_datasets(self,file_name):
        data_transform = transform.transform()
        train_loader, dataset = get_loader(
            root_folder="./",
            annotation_file=file_name,
            transform=data_transform,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        return train_loader, dataset
        
    def setting_and_training(self, train_loader, dataset):
        torch.backends.cudnn.enabled = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vocab_size = len(dataset.vocab)
        step = 0
        model = CNNtoLSTM(self.embed_size, self.hidden_size, vocab_size, self.num_layers).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
        
        optimizers = {
            'Adam': optim.Adam,
            'Adagrad': optim.Adagrad,
            'AdamW': optim.AdamW,
            'Adadelta': optim.Adadelta,
            'Adamax': optim.Adamax,
            'NAdam': optim.NAdam,
            'RAdam': optim.RAdam,
            'RMSprop': optim.RMSprop
        }

        if self.opti in optimizers:
            optimizer = optimizers[self.opti](model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.opti} not recognized")

        opti_name = f"{self.opti}_{self.num_layers}"
        scheduler = StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.gamma)
    
        # Only finetune the CNN
        for name, param in model.encoderCNN.vit.named_parameters():
            if "heads.weight" in name or "heads.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
                
        patience = 20
        early_stopping = EarlyStopping(opti_name, patience, verbose=True)
        
        model.train()
        
        for epoch in range(self.num_epochs):
            print(f'>>>>>> epoch: {epoch} >>>>>>')
        
            for idx, (imgs, captions) in tqdm(enumerate(train_loader), 
                                              total=len(train_loader), leave=False):
                imgs = imgs.to(device)
                captions = captions.to(device)
        
                outputs = model(imgs, captions[:-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
                step += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            early_stopping(loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            torch.cuda.empty_cache()
            print(f'loss: {loss.item()}')

        return model
    
    def save(self, model):
        if not os.path.isdir('finish_model'):
            os.mkdir('finish_model')
        model_path = f'finish_model/{self.opti}_{self.num_layers}.pth'
        torch.save(model.state_dict(), model_path)
        torch.cuda.empty_cache()
