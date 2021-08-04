from pytorch_lightning import trainer
from pytorch_lightning.trainer.trainer import Trainer
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Fully connected neural network with one hidden layer
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)  
    
    def training_step(self,batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        return {'loss':loss}

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, num_workers=4)

        return train_loader
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        return {'val_loss':loss}

    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=False, 
                                           transform=transforms.ToTensor())

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False, num_workers=4)

        return val_loader
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss':avg_loss}
        

if __name__ == '__main__':
    trainer = Trainer(fast_dev_run = False, log_every_n_steps=1,max_epochs=num_epochs)
    model = LitNeuralNet(input_size, hidden_size, num_classes)

    trainer.fit(model)

