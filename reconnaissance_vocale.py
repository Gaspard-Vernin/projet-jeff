import torch 
import numpy as np 
import scipy as sp
def passe_bande(signal):
    s = np.fft.fft(signal)
    s[:20]=0 
    s[20000:]=0
    return np.fft.ifft(s)
class Network(torch.nn.Module):
    def __init__(self,dim_sortie=10,taille_out_channels=64):
        super().__init__() 
        self.dim_sortie=dim_sortie
        self.conv_lay1 = torch.nn.Conv2d(in_channels=1,out_channels=taille_out_channels//2,padding=1,kernel_size=3)
        self.norm1 = torch.nn.BatchNorm2d(num_features=taille_out_channels//2)
        self.conv_lay2 = torch.nn.Conv2d(in_channels=taille_out_channels//2,out_channels=taille_out_channels,padding=1,kernel_size=3)
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
        x=torch.flatten(x,start_dim=1)
        x=self.rel1(self.fc1(x))
        x=self.rel2(self.fc2(x))
        return self.fc3(x) 
    def backprop(self, resultats,output):
        if(len(resultats)!=output.shape[0]):
            raise ValueError(f"on a pas les resultats pour toutes les output : dim resultats = {len(resultats)}, dim output = {output.shape[0]}")
        res = torch.zeros(output.shape[0],self.dim_sortie)
        for i in range(len(resultats)):
            res[i,resultats[i]]=1 
        loss = torch.nn.MSELoss()
        val_loss = loss(output,res)
        self.optimizer.zero_grad()
        val_loss.backward()
        self.optimizer.step()
def train(dataset,nb_mots_differents_dataset,nb_epoch=3,size_mini_batch=32):
    """créer et entraine reseau de detections de mots,
        dataset est de la forme [(signal,numero du mot représenté)]
    """
    net = Network(dim_sortie=nb_mots_differents_dataset)
    for _ in range(nb_epoch): 
        i=0
        while i<len(dataset):
            i_temp = 0
            batch=[]
            nums=[]
            while i+i_temp<len(dataset) and i_temp<size_mini_batch:
                batch.append(torch.tensor(dataset[i+i_temp][0]).unsqueeze(0))
                nums.append(dataset[i+i_temp][1])
                i_temp+=1 
            i+=i_temp
            batch=torch.stack(batch)
            forw = net(torch.tensor(batch))
            net.backprop(nums,forw)
    return net
def test_forward():
    signal1 = np.zeros(20000)
    signal1=passe_bande(signal1)
    _,_,signal1=sp.signal.stft(signal1)
    signal1=np.power(np.abs(signal1),2,dtype=np.float32)
    #on unsqueeze pour créer la dimension des canaux.
    dataset = [(signal1,0)]*39
    train(dataset=dataset,nb_mots_differents_dataset=1)
test_forward()