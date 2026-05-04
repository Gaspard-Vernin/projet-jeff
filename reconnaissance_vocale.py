import torch 
import numpy as np
import matplotlib 

class Network(torch.nn.Module):
    def __init__(self,dim_sortie=10,taille_out_channels=64):
        super().__init__() 
        self.dim_sortie=dim_sortie
        self.conv_lay1 = torch.nn.Conv2d(in_channels=1,out_channels=taille_out_channels//2,padding=1,kernel_size=3)
        self.norm1 = torch.nn.BatchNorm2d(num_features=taille_out_channels//2)
        self.conv_lay2 = torch.nn.Conv2d(in_channels=1,out_channels=taille_out_channels,padding=1,kernel_size=3)
        self.norm2=torch.nn.BatchNorm2d(num_features=taille_out_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = torch.nn.Linear(taille_out_channels,taille_out_channels)
        self.rel1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(taille_out_channels,taille_out_channels//2)
        self.rel2=torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(taille_out_channels//2,dim_sortie)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-3)
    def forward(self,input):
        x=self.norm1(self.conv_lay1(input))
        x=self.norm2(self.conv_lay2(x))
        x=self.pool(x)
        x=self.gap(x)
        x=torch.flatten(x)
        x=self.rel1(self.fc1(x))
        x=self.rel2(self.fc2(x))
        return self.fc3(x) 
    def backprop(self, resultat,output):
        res = torch.zeros(self.dim_sortie)
        res[resultat]=1 
        loss = torch.nn.MSELoss()
        val_loss = loss(output-resultat)
        self.optimizer.zero_grad()
        val_loss.backward()
        self.optimizer.step()
def train(dataset,nb_mots_differents_dataset,nb_epoch=3):
    """créer et entraine reseau de detections de mots,
        dataset est de la forme [(signal,numero du mot représenté)]
    """
    net = Network(dim_sortie=nb_mots_differents_dataset)
    for _ in range(nb_epoch):
        for (signal,num) in dataset:
            forw = net(signal)
            net.backprop(num,forw)
    return net
        
