import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import numpy as np
import math
# from d2l import torch as d2l
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import sys
from typing import Iterable, Tuple, List, Union

sys.path.append(r'D://projectcode//python//rkn_share-master//util')

from ConfigDict import ConfigDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#随机种子
SEED=1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False



class MLP(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(MLP, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.fc_1 =  nn.Linear(input_size, input_size) #fully connected 1
        self.fc_2 =  nn.Linear(input_size, 1) #fully connected 1
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self,x):
        x = x.view(-1,  self.input_size) 
        # print("re X:",x)
        x = self.fc_1(x)
        x = self.relu(x)
        out = self.fc_2(x) #Final Output
        return out

class simpleRNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(simpleRNN, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #GRU
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
        # Propagate input through LSTM
        output, (hn) = self.rnn(x, (h_0)) #lstm with input, hidden, and internal state
        # output, (hn, cn) = self.lstm(x, (hn, cn))
        # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        # out = self.relu(hn)
        # out = self.fc_1(out) #first Dense
        # out = self.relu(out) #relu
        # out = self.fc(out) #Final Output
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next

        out = self.relu(hn)
        out = self.fc(hn) #Final Output
        return out





class vanillaTransformer(nn.Module):
 
    def __init__(self, batchSize=128,sequenceLength=10, ntoken=1, ninp=35, d_input=12, nhead=1, nhid=35, nlayers=1, dropout=0.2,batchSizeVal=1):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
        super(vanillaTransformer, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=12, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=12,
                            dropout=dropout, activation='relu', custom_encoder=None, custom_decoder=None,batch_first=True)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=60, nhead=3,batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self._embedding = nn.Linear(12, 12)
        # self._embeddingDecoder = nn.Linear(d_input, 35)

        self.positionalEncoding = (torch.arange(0,sequenceLength)/(0.5*sequenceLength)).repeat(12,1).T.repeat(batchSize,1,1).to(device)

        self.end = nn.Linear(12,1)
 
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
    def forward(self,x):
 
        # src.shape[1]
        # pos = (src[:, :, 0:7] - poseMean) / poseSTD
        # vel = (src[:, :, 7:14] - velocityMean) / velocitySTD
        # acc = (src[:, :, 14:21] - accelerationMean) / accelerationSTD
        # neEst = (src[:, :, 21:28] - torqueMean) / torqueSTD #刚体动力学计算得到的扭矩
        # turn = (src[:, :, 28:35] - 0) / 50
 
 
        src2 = x
        # K = src.shape[1]
 
        # Embeddin module
        encoding = self._embedding(src2)
        encoding = torch.add(encoding, self.positionalEncoding[:encoding.shape[0],:,:])

        output = self.transformer(encoding,src2)

        output = self.end(output[:,-1,:])

        #去标准化操作
        # output = torch.mul(output, torqueSTD.repeat(batchSize,1))
        # output = torch.add(output, torqueMean.repeat(batchSize,1))
 
        # output = torch.add(src[:,-1, 21:21+7], output)

        return output


class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #GRU
        # self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
        # Propagate input through LSTM
        output, (hn) = self.gru(x, (h_0)) #lstm with input, hidden, and internal state
        # output, (hn, cn) = self.lstm(x, (hn, cn))
        # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        # out = self.relu(hn)
        # out = self.fc_1(out) #first Dense
        # out = self.relu(out) #relu
        # out = self.fc(out) #Final Output
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        # out = self.relu(hn)
        # out = self.fc_1(out) #first Dense
        # out = self.relu(hn) #relu
        out = self.selu(hn)
        out = self.fc(hn) #Final Output
        return out

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer，单向网络层配置
        self.relu = nn.ReLU()
        self.selu =nn.SELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        hn = self.selu(hn)
        out = self.fc(hn) #Final Output
        return out



class ContactComp_Implicated(nn.Module):
    def __init__(self, train_joint ,num_classes, input_size, hidden_size, num_layers, seq_length, dof):

        super(ContactComp_Implicated, self).__init__() 
        self.num_classes = num_classes #number of classes 
        self.num_layers = num_layers #number of layers 
        self.input_size = input_size #input size 
        self.hidden_size = hidden_size #hidden state 
        self.seq_length = seq_length #sequence length 
        self.dof = dof 
                
        self.lstm_delta3 = nn.LSTM(input_size= 1, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm 
    
        self.fc1 = nn.Linear(self.dof,1) 
        self.fc3 = nn.Linear(1*self.hidden_size,1) 
        self.fn = input_size 

        self.spatial_weight1 = nn.Linear(self.dof,self.dof) 
        self.hn_temporal_weight2 = nn.Linear(self.seq_length,1*self.hidden_size)
        
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid() 
        self.selu = nn.SELU() 
    
    def forward(self,x,train_joint,future_data,oldref): 
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置 
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置 
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置 

        h_3 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置 
        c_3 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        x = x[:,:,:2*self.dof] 

        q = x[:,:,:1*self.dof] 
        v = x[:,:,1*self.dof:2*self.dof] 


        spatial_attention1 = self.spatial_weight1(q[:,-1,:]) #q[:,-1,:] 
        spatial_attention1 = self.selu(spatial_attention1) 
        spatial_scores1 = nn.functional.softmax(spatial_attention1, dim=-1) 
        testq = torch.mul(q[:,-1,:],spatial_scores1) #q[:,-1,:] 
        # spatial_scores1 = torch.zeros(q[:,-1,:].shape) 

        output_d2, (hnd2, cnd2) = self.lstm_delta3(v[:,:,train_joint].unsqueeze(2), (h_3, c_3)) 
        hnd2 = hnd2.squeeze(0) 

        temporal_weight2 = self.hn_temporal_weight2(v[:,:,train_joint]) # torch.cat((q[:,-1,:],v[:,:,train_joint]),dim=1)
        temporal_weight2 = self.selu(temporal_weight2) 
        temporal_scores2 = nn.functional.softmax(temporal_weight2, dim=-1) 
        hnd2 = torch.mul(hnd2,temporal_scores2) 
        # temporal_scores2 = torch.zeros(v[:,:,train_joint].shape) 

        momentum_power = self.fc1(self.selu(testq)) 
        momentum_power1 = self.fc3(self.selu(hnd2)) 

        out1 = momentum_power1+torch.mul(momentum_power1,momentum_power)#+torch.mul(momentum_power1,momentum_power)#+momentum_power##torch.mul(momentum_power1,momentum_power)+

        return out1,spatial_scores1,temporal_scores2,momentum_power1,momentum_power1#torch.cat((v[:,-1,:]), dim=1) 
