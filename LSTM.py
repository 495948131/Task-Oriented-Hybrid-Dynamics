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





class LSTM_withR(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_withR, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer，单向网络层配置
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.temporal_weight = nn.Linear(hidden_size,hidden_size)
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    
    def forward(self,x,r):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        test = torch.cat((x,r),dim=2)
        output, (hn, cn) = self.lstm(test, (h_0, c_0)) #lstm with input, hidden, and internal state
        
        temporal_weight = self.temporal_weight(output)
        temporal_weight = self.sigmoid(temporal_weight)
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        hn = hn.view(-1, self.hidden_size)
        hn = self.relu(hn)
        out = self.fc(hn) #Final Output
        return out,temporal_scores

class LSTM_VAE(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,latent_dims):
        super(LSTM_VAE, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.latent_dims = latent_dims #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        # self.lstm_flip = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #                   num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        # self.fc = nn.Linear(hidden_size+latent_dims, num_classes) #fully connected last layer，单向网络层配置
        self.fc1 = nn.Linear(hidden_size,6)
        self.fc2 = nn.Linear(latent_dims+6,num_classes)
        # self.fc3 = nn.Linear(12,1)
        
        #注意力机制
        # self.num_attention_heads = 1
        # self.attention_head_size = int((hidden_size*2+latent_dims) / self.num_attention_heads)
        # self.all_head_size = hidden_size*2+latent_dims

        # self.query = nn.Linear(2*hidden_size+latent_dims,2*hidden_size+latent_dims)
        # self.key = nn.Linear(2*hidden_size+latent_dims,2*hidden_size+latent_dims)
        # self.value = nn.Linear(2*hidden_size+latent_dims,2*hidden_size+latent_dims)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,1,2)
    
    def forward(self,x,latent):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        # h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        # c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        # x_flip = torch.flip(x,dims = [1])
        # output_filp, (hn_flip, cn_flip) = self.lstm_flip(x_flip, (h_1, c_1)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        # hn_flip = hn_flip.view(-1, self.hidden_size)
        # hn_cat = torch.cat([hn,hn_flip],1)
        latent = latent.reshape(-1, self.latent_dims)
        out2 = torch.cat([hn,latent],1)
        
        #注意力机制调试
        # mixed_query_layer = self.query(out2)
        # mixed_key_layer = self.key(out2)
        # mixed_value_layer = self.value(out2)
        
        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # context_layer = torch.matmul(attention_probs, value_layer)
        
        # print(attention_scores.size())
        
        out1 = self.fc1(hn) #Final Output
        out1 = self.tanh(out1)
        out2 = torch.cat([out1,latent],1)
        out2 = self.fc2(out2)
        out2 = self.tanh(out2)
        # out = self.fc3(out2)
        return out2
    
    
class LSTM_VAE_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,latent_dims):
        super(LSTM_VAE_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.latent_dims = latent_dims #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(300,1)
        # self.fc2 = nn.Linear(latent_dims+6,num_classes)
        
        self.W_k = nn.Linear(latent_dims, hidden_size, bias=False)
        self.W_q = nn.Linear(14, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def masked_softmax(X, valid_lens):
        # X:3D张量，valid_lens:1D或2D张量
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    
    def forward(self,x,latent):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        pca_key = np.zeros((x.shape[0],x.shape[1],1))
        print(pca_key.shape)
        for i in range(1,x.shape[0]):
            pca = PCA(n_components=1,copy=True)
            pca_key[i,:,:] = pca.fit_transform(x[0,:,:].data.cpu().numpy())
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        latentx = latent
        hn = hn.view(-1, self.hidden_size)

        latent = latent.reshape(-1, self.latent_dims)
        out2 = torch.cat([hn,latent],1)

        queries, keys = self.W_q(x), self.W_k(latentx)
        # print(queries.size())
        # print(keys.size())
        # print(queries.unsqueeze(2).size())
        # print(keys.unsqueeze(1).size())
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        scores = nn.functional.softmax(scores, dim=-1)
        attention = torch.bmm(scores, hn.unsqueeze(1))
        
        attention = attention.reshape(-1, 300)
        # print(attention.size())
        
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        out1 = self.fc1(attention) #Final Output
        # out1 = self.tanh(out1)
        # out2 = torch.cat([out1,latent],1)
        # out2 = self.fc2(out2)
        # out2 = self.tanh(out2)
        return out1


class LSTM_PCA_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(300,1)
        
        self.W_k1 = nn.Linear(14, hidden_size, bias=False)
        self.W_q = nn.Linear(14, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def masked_softmax(X, valid_lens):
        # X:3D张量，valid_lens:1D或2D张量
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        pca_key = np.zeros((x.shape[0],14,1))
        for i in range(1,x.shape[0]):
            pca = PCA(n_components=1,copy=True)
            x_test = x.transpose(1,2)
            pca.fit_transform(x_test[i,:,:].data.cpu().numpy())
            # print(pca.fit_transform(x_test[i,:,:].data.cpu().numpy()).shape)
            pca_key[i,:,:] = pca.fit_transform(x_test[i,:,:].data.cpu().numpy())
            # pca.fit_transform(x[i,:,:].data.cpu().numpy())
            print(pca.explained_variance_ratio_)
            # quit()
        pca_key = torch.tensor(pca_key,dtype=torch.float).to(device)
        # print(pca_key.shape)
        # quit()
        pca_key = pca_key.transpose(1,2)
        keys2 = self.W_k1(pca_key)
        queries2 = self.W_q(x)
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        # queries, keys = self.W_q(x), self.W_k(latentx)
        # print(queries.size())
        # print(keys.size())
        # print(queries.unsqueeze(2).size())
        # print(keys.unsqueeze(1).size())
        features = queries2.unsqueeze(2) + keys2.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        scores = nn.functional.softmax(scores, dim=-1)
        # print(scores)
        attention = torch.bmm(scores, hn.unsqueeze(1))
        
        attention = attention.reshape(-1, 300)
        # print(attention.size())
        
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        out1 = self.fc1(attention) #Final Output
        return out1



class LSTM_Spatial_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Spatial_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(10,1)
        self.spatial_weight = nn.Linear(14,14)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(scores,x)
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        hn = self.tanh(hn)
        out1 = self.fc1(hn) #Final Output
        return out1,scores
    

class LSTM_Spatial_Attention_from_output(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Spatial_Attention_from_output, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(10,1)
        self.spatial_weight = nn.Linear(14,14)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(scores,x)
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        hn = self.tanh(hn)
        out1 = self.fc1(hn) #Final Output
        return out1,scores


class LSTM_Time_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Time_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(300,1)
        self.time_weight = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.1)
    
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn_attention = self.time_weight(output)
        hn_attention = self.relu(hn_attention)
        hn_scores = nn.functional.softmax(hn_attention, dim=-1)
        hn_weight = torch.mul(hn_scores,output)
        hn_weight = self.tanh(hn_weight)
        hn_weight=hn_weight.view(-1, 300)
        out1 = self.fc1(hn_weight) #Final Output
        return out1,hn_scores

class LSTM_Spatial_Time_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Spatial_Time_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(100,1)
        self.spatial_weight = nn.Linear(14,14)
        self.time_weight = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.1)
    
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(scores,x)
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn_attention = self.time_weight(output)
        hn_scores = self.relu(hn_attention)
        hn_scores = nn.functional.softmax(hn_scores, dim=-1)
        hn_weight = torch.mul(hn_scores,output)
        hn_weight = self.tanh(hn_weight)
        hn_weight=hn_weight.view(-1, self.hidden_size*self.hidden_size)
        out1 = self.fc1(hn_weight) #Final Output
        return out1,scores

#PCA方法似乎在强化一些不该强化的特征，PCA方法其实是对时域尺度上的特征进行了提取(从结果分析)
#但是对于机械臂而言，数据集在时域尺度的上的特征总是不是很明确，会造成训练过程不收敛和过拟合等问题，进一步提取或强化时域特征不利于
#提高模型精度和泛化性。
#关于时域特征问题如果可以分析清楚，应该能够进一步的加强注意力机制的理论分析
class LSTM_PCA_Spatial_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_Spatial_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        self.fc1 = nn.Linear(10,1)
        self.spatial_weight = nn.Linear(7,10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        pca_key = np.zeros((x.shape[0],14,7))
        for i in range(1,x.shape[0]):
            pca = PCA(n_components=7,copy=True)
            x_pca = x.transpose(1,2)
            pca.fit_transform(x_pca[i,:,:].data.cpu().numpy())
            pca_key[i,:,:] = pca.fit_transform(x_pca[i,:,:].data.cpu().numpy())
        pca_key_train = torch.tensor(pca_key,dtype=torch.float).to(device)
        spatial_attention = self.spatial_weight(pca_key_train)
        spatial_attention = self.sigmoid(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        scores = scores.transpose(1,2)
        x_weight = torch.mul(scores,x)
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        hn = self.tanh(hn)
        out1 = self.fc1(hn) #Final Output
        return out1,scores
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # 定义多个注意力头
        self.attention_heads = nn.ModuleList([SelfAttention(input_dim, self.head_dim) for _ in range(num_heads)])

        # 定义线性层用于将多个注意力头的输出进行投影
        self.projection = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 将输入张量按注意力头维度分拆，并分别传入不同的注意力头进行计算
        head_outputs = [attention_head(x) for attention_head in self.attention_heads]  # shape: [(batch_size, seq_len, head_dim)] * num_heads
        
        # 将多个注意力头的输出张量在头维度上进行拼接
        concatenated_head_outputs = torch.cat(head_outputs, dim=-1)  # shape: (batch_size, seq_len, input_dim)
        
        # 对拼接后的张量进行线性变换投影
        projected_output = self.projection(concatenated_head_outputs)  # shape: (batch_size, seq_len, input_dim)
        
        return projected_output



class SelfAttention(nn.Module):
    def __init__(self, input_dim, head_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim

        # 定义线性层用于计算查询、键和值的权重
        self.query = nn.Linear(input_dim, head_dim)
        self.key = nn.Linear(input_dim, head_dim)
        self.value = nn.Linear(input_dim, head_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 将输入张量通过线性变换得到查询、键和值
        Q = self.query(x)  # shape: (batch_size, seq_len, head_dim)
        K = self.key(x)  # shape: (batch_size, seq_len, head_dim)
        V = self.value(x)  # shape: (batch_size, seq_len, head_dim)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_probs = torch.softmax(attention_scores, dim=-1)  # shape: (batch_size, seq_len, seq_len)
        
        # 计算加权和
        attended_values = torch.matmul(attention_probs, V)  # shape: (batch_size, seq_len, head_dim)
        
        return attended_values

class LSTM_PCA_Time_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_Time_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        
        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,hidden_size)#time_size*hidden_size
        self.fn = input_size
        self.spatial_weight = nn.Linear(self.fn,self.fn) 
        self.spatial_weight1 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight2 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight3 = nn.Linear(self.fn,self.fn)
        self.spatial_weight4 = nn.Linear(self.fn,self.fn)
        self.spatial_weight5 = nn.Linear(self.fn,self.fn)
        
        
        self.temporal_weight = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight2 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight3 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight4 = nn.Linear(hidden_size,hidden_size)
        self.temporal_weight5 = nn.Linear(hidden_size,hidden_size)
        self.temporal_weight6 = nn.Linear(hidden_size,hidden_size)
        
        self.vel_weight = nn.Linear(self.seq_length,hidden_size)
       
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self,x,train_joint):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        result = self.vel_weight(x[:,:,(7+train_joint)])
        
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.spatial_weight2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.spatial_weight3(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.spatial_weight4(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.spatial_weight5(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
        
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) 
        
        temporal_weight = self.temporal_weight(output)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight1(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight2(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight3(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight4(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight5(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight6(temporal_weight)
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output) 
        
        temporal_outputs = temporal_outputs+output
        temporal_mean = torch.mean(temporal_outputs, dim=1)
        hn = torch.squeeze(hn,dim=0)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(temporal_outputs.shape[0],5)
        hn_q = danweiT-result
        to_q = result
        short_outputs = torch.mul(hn_q,hn)
        temporal_mean = torch.mul(to_q,temporal_mean)
        outputs = torch.cat((temporal_mean,short_outputs),dim=1)
        outputs = self.relu(outputs)
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs)
        out1 = self.fc(outputs) 
        return out1,spatial_scores

class LSTM_PCA_Time_Attention_comp1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_Time_Attention_comp1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        
        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,hidden_size)#time_size*hidden_size
        self.fn = input_size
        self.spatial_weight = nn.Linear(self.fn,self.fn) 
        self.spatial_weight1 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight2 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight3 = nn.Linear(self.fn,self.fn)
        # self.spatial_weight4 = nn.Linear(self.fn,self.fn)
        # self.spatial_weight5 = nn.Linear(self.fn,self.fn)
        
        
        # self.temporal_weight = nn.Linear(hidden_size,hidden_size) 
        # self.temporal_weight1 = nn.Linear(hidden_size,hidden_size) 
        # self.temporal_weight2 = nn.Linear(hidden_size,hidden_size) 
        # self.temporal_weight3 = nn.Linear(hidden_size,hidden_size) 
        # self.temporal_weight4 = nn.Linear(hidden_size,hidden_size)
        # self.temporal_weight5 = nn.Linear(hidden_size,hidden_size)
        # self.temporal_weight6 = nn.Linear(hidden_size,hidden_size)
        
        self.vel_weight = nn.Linear(self.seq_length,2*hidden_size)
       
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self,x,train_joint):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        result = self.vel_weight(x[:,:,(7+train_joint)])
        
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        # spatial_attention = self.spatial_weight1(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        # spatial_attention = self.spatial_weight2(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        # spatial_attention = self.spatial_weight3(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        # spatial_attention = self.spatial_weight4(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        # spatial_attention = self.spatial_weight5(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        
        # temporal_weight = self.temporal_weight(hn)
        # temporal_weight = self.relu(temporal_weight)
        # temporal_weight = self.temporal_weight1(temporal_weight)
        # temporal_weight = self.relu(temporal_weight)
        # temporal_weight = self.temporal_weight2(temporal_weight)
        # temporal_weight = self.relu(temporal_weight)
        # temporal_weight = self.temporal_weight3(temporal_weight)
        # temporal_weight = self.relu(temporal_weight)
        # temporal_weight = self.temporal_weight4(temporal_weight)
        # temporal_weight = self.relu(temporal_weight)
        # temporal_weight = self.temporal_weight5(temporal_weight)
        # temporal_weight = self.relu(temporal_weight)
        # temporal_weight = self.temporal_weight6(temporal_weight)
        # temporal_weight = self.relu(temporal_weight) 
        # temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        # temporal_outputs = torch.mul(temporal_scores,hn) 
        
        # temporal_outputs = temporal_outputs + hn
        temporal_outputs = torch.squeeze(output,dim=0)
        outputs = self.relu(temporal_outputs)
        out1 = self.fc(outputs) 
        return out1,spatial_scores


class LSTM_st_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=1*input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        # self.lstm_future = nn.LSTM(input_size=9, hidden_size=hidden_size,
        #                   num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size 
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size 
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((4,8))
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.spatial_weight = nn.Linear(1*self.input_size,1*self.input_size) 
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,2*hidden_size)
        self.res = nn.Linear(1*input_size,2*hidden_size)
        # self.res = nn.Linear(2*input_size,2*hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        # self.batchnorm = nn.BatchNorm1d(seq_length)
        # self.dropout = nn.Dropout(0.01)
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        # h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        # c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        # vel_x = torch.cat((x, future_data), dim=1) 
        vel_x = x
        velocity = vel_x[:,:,train_joint+self.dof] 
        result = self.vel_weight(velocity) 
        result = self.selu(result) 

        # x = torch.cat((x, future_data), dim=2) 
        # x = self.dropout(x)
        x = x[:,:,:2*self.dof]
        # x = self.batchnorm(x)
        spatial_attention = self.spatial_weight(x) 
        spatial_attention = self.selu(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,x) 
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x_weight) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:], (h_0, c_0)) #self.input_size
        
        output = self.sigmoid(output) 
        output = output + res 
            
        hn = output[:,-1,:]
        hn = self.sigmoid(hn)  
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.relu(hn_weight) 
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1) 
        hn = torch.mul(hn_temporal_scores,hn)   
        
        output = torch.mean(output, dim=1)
        output = self.sigmoid(output)  
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        temporal_scores = torch.cat((temporal_scores,hn_temporal_scores),dim=1)
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        hn_temporal_scores = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1)   
        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs) 
        outputs = self.selu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,temporal_scores,to_q


class LSTM_st_noCNN_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_noCNN_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        # self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight = nn.Linear(1*self.input_size,1*self.input_size) 

        
        # self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        # self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,2*hidden_size)

        self.res = nn.Linear(input_size,2*hidden_size)
        # self.res = nn.Linear(2*input_size*seq_length,2*hidden_size*6)

        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定
        # vel_x = torch.cat((x, future_data), dim=1) 
        velocity = x[:,:,self.dof+train_joint]
        result = self.vel_weight(velocity) 
        result = self.selu(result)

        # x = torch.cat((x, future_data), dim=2) 
        x = x[:,:,:2*self.dof]
        # spatial_attention = self.spatial_weight(x) 
        # spatial_attention = self.sigmoid(spatial_attention) 
        # spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        # x_weight = torch.mul(spatial_scores,x) 
        
        #无空间注意力可选项
        spatial_scores = torch.zeros(x[:,:,:2*self.dof].shape)
        x_weight = x+torch.zeros(x[:,:,:2*self.dof].shape)

        res = self.res(x_weight)          

        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        
        output = self.sigmoid(output)
        output = output+ res 
        
        hn = output[:,-1,:] 
        hn = self.sigmoid(hn) 
        # hn_weight = self.hn_temporal_weight1(hn) 
        # hn_weight = self.relu(hn_weight) 
        # hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1) 
        # hn = torch.mul(hn_temporal_scores,hn) 

        output = torch.mean(output,dim=1)
        output = self.sigmoid(output)
        # temporal_weight = self.temporal_weight(output) 
        # temporal_weight = self.relu(temporal_weight) 
        # temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        # temporal_outputs = torch.mul(temporal_scores,output)
        
        #无时间注意力机制
        temporal_outputs = output+torch.zeros(output.shape)
        temporal_scores = torch.zeros(output.shape)
        hn = output+torch.zeros(hn.shape)
        hn_temporal_scores = torch.zeros(hn.shape)
        
        temporal_scores = torch.cat((temporal_scores,hn_temporal_scores),dim=0)
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        hn_temporal_scores = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs) 
        outputs = hn
        outputs = self.selu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,hn_temporal_scores,to_q

class LSTM_onlyt_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_onlyt_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=1*input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,2*hidden_size)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((4,8))
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.res = nn.Linear(self.input_size,2*hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        vel_x = x
        velocity = vel_x[:,:,self.dof+train_joint]
        result = self.vel_weight(velocity) 
        result = self.selu(result)

        # x = torch.cat((x, future_data), dim=2) 
        spatial_scores = torch.zeros(x.shape)

        x = x.permute(0, 2, 1) 
        x = self.conv(x) 
        x = x.permute(0, 2, 1) 
        
        res = self.res(x)          

        output, (hn, cn) = self.lstm(x[:,:,:], (h_0, c_0)) 
        
        output = self.sigmoid(output) 
        output = output+ res 

        hn = output[:,-1,:] 
        hn = self.sigmoid(hn) 
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.relu(hn_weight) 
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1) 
        hn = torch.mul(hn_temporal_scores,hn)
         
        output = torch.mean(output,dim=1)
        output = self.sigmoid(output)
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        temporal_scores = torch.cat((temporal_scores,hn_temporal_scores),dim=0)
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        hn_temporal_scores = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs) 
        outputs = self.selu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,hn_temporal_scores,to_q


class LSTM_onlys_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_onlys_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=1*input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((4,8))
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.spatial_weight = nn.Linear(1*self.input_size,1*self.input_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,2*hidden_size)

        self.res = nn.Linear(self.input_size,2*hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        

        #位移值根据机械臂自由度确定
        vel_x = x
        velocity = vel_x[:,:,self.dof+train_joint]
        result = self.vel_weight(velocity) 
        result = self.selu(result)

        
        spatial_attention = self.spatial_weight(x) 
        spatial_attention = self.sigmoid(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,x) 
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1)

        res = self.res(x_weight)          

        output, (hn, cn) = self.lstm(x_weight[:,:,:], (h_0, c_0)) 
        
        output = self.sigmoid(output)
        output = output+ res 
        
        to_q = result
        hn = output[:,-1,:] 
        hn_temporal_scores = torch.zeros(hn.shape)
        outputs = self.selu(hn) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,hn_temporal_scores,to_q

class LSTM_st_Attention_no_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_Attention_no_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        # self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #                   num_layers=num_layers, batch_first=True,bidirectional=False) #lstm

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((4,8))
            nn.MaxPool1d(kernel_size=2, stride=2) 
        )

        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,1*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        
        self.temporal_weight = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_test = nn.Linear(self.hidden_size*6,self.hidden_size)
        
        self.hn_temporal_weight1 = nn.Linear(self.hidden_size,self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,hidden_size)

        self.res = nn.Linear(12,hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        #位移值根据机械臂自由度确定
        velocity = x[:,:,self.dof+train_joint]
        result = self.vel_weight(velocity) 
        result = self.selu(result)

        old_x = x[:,:,:2*self.dof]
        spatial_attention = self.spatial_weight(old_x) 
        spatial_attention = self.sigmoid(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,old_x) 

        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 

        res = self.res(x_weight)
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:], (h_0, c_0)) 
       
        output = self.sigmoid(output) 
        output = output+ res 
        
        hn = output[:,-1,:] 
        hn = self.sigmoid(hn) 
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.relu(hn_weight) 
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1) 
        hn = torch.mul(hn_temporal_scores,hn) 
        
        output = torch.mean(output,dim=1) 
        output = self.sigmoid(output) 
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        temporal_scores = torch.cat((temporal_scores,hn_temporal_scores),dim=0)
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        hn_temporal_scores = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs) 
        outputs = self.selu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,hn_temporal_scores,to_q

class LSTM_st_div_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_div_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(4*hidden_size,1)#time_size*hidden_size
        # self.fc1 = nn.Linear(2*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        # self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight1 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight2 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight3 = nn.Linear(2*self.input_size,2*self.input_size) 
        
        self.spatial_weight = nn.Linear(self.dof,self.dof) 
        self.spatial_weight1 = nn.Linear(self.dof,self.dof) 
        self.spatial_weight2 = nn.Linear(self.dof,self.dof) 
        self.spatial_weight3 = nn.Linear(self.dof,self.dof) 
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
        
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight5 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        self.vel_weight = nn.Linear(20,2*hidden_size)
        self.res = nn.Linear(16*self.input_size,80)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(8)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        vel_x = torch.cat((x, future_data), dim=1) 
        # result = self.vel_weight(vel_x[:,:,(self.dof+train_joint)]) 
        # result = vel_x[:,:,self.dof:2*self.dof]
        velocity = vel_x[:,:,self.dof+train_joint]
        position = vel_x[:,:,train_joint]
        # result = torch.cat((position, velocity), dim=1) 
        result = self.vel_weight(velocity) 
        result = self.relu(result)
        #分散
        velocity = vel_x[:,:,self.dof:2*self.dof]
        position = vel_x[:,:,0:self.dof]
        # velocity_norm = self.batchnorm1(velocity)
        velocity_attention = self.spatial_weight(velocity) 
        velocity_attention = self.sigmoid(velocity_attention) 
        
        velocity_attention = self.spatial_weight1(velocity_attention) 
        velocity_attention = self.sigmoid(velocity_attention) 
        velocity_scores = nn.functional.softmax(velocity_attention, dim=-1) 
        velocity_weight = torch.mul(velocity_scores,velocity) 
        
        # position_norm = self.batchnorm1(position)
        position_attention = self.spatial_weight2(position) 
        position_attention = self.sigmoid(position_attention) 
        
        position_attention = self.spatial_weight3(position_attention) 
        position_attention = self.sigmoid(position_attention) 
        position_scores = nn.functional.softmax(position_attention, dim=-1) 
        position_weight = torch.mul(position_scores,position) 
        
        spatial_scores = torch.cat((position_scores, velocity_scores), dim=2) 
        x_weight = torch.cat((position_weight,velocity_weight), dim=2)
        
        #卷积
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = x_weight.reshape(-1,16*self.input_size) 
        res = self.res(res) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:8,:], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,8:,:],dims=[1]), (h_1, c_1))#torch.flip(x_weight[:,8:,:],dims=[2]) 
        
        output = torch.cat((output, output1), dim=2) 
        res = res.reshape(-1,8,10) 
        output = output+ res 
        
        hn = output[:,-1,:] 
        # hn = self.sigmoid(hn) 
        
        output = torch.mean((output[:,:,:]), dim=1)
        # output = self.sigmoid(output) 
        # norm_output = self.batchnorm(output) 
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight1(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight2(temporal_weight) 
        temporal_weight = self.selu(temporal_weight)
        
        temporal_weight = self.temporal_weight3(temporal_weight) 
        temporal_weight = self.selu(temporal_weight)
        
        # norm_hn = self.batchnorm(hn)
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight2(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight3(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight4(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,temporal_scores,to_q


class LSTM_SNet_t_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_SNet_t_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(4*hidden_size,1)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2*input_size, out_channels=2*input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        self.snet1 = nn.Linear(2*input_size, 2*input_size, bias=False)
        self.snet2 = nn.Linear(2*input_size, 2*input_size, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight5 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        self.vel_weight = nn.Linear(20,2*hidden_size)
        self.res = nn.Linear(2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(40)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(10)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        vel_x = torch.cat((x, future_data), dim=1) 
        # result = self.vel_weight(vel_x[:,:,(self.dof+train_joint)]) +
        # result = vel_x[:,:,self.dof:2*self.dof]
        velocity = vel_x[:,:,self.dof+train_joint]
        # position = vel_x[:,:,train_joint]
        # result = torch.cat((position, velocity), dim=1) 
        result = self.vel_weight(velocity) 
        result = self.relu(result)
        
        x = torch.cat((x, future_data), dim=2)
        
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = self.snet1(avg_pool) 
        avg_pool = self.relu(avg_pool)
        avg_pool = self.snet2(avg_pool) 
        
        max_pool = self.snet1(max_pool) 
        max_pool = self.relu(max_pool)
        max_pool = self.snet2(max_pool) 

        spatial_attention = avg_pool + max_pool
        spatial_scores = self.sigmoid(spatial_attention)
        x_weight = x * spatial_scores
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        res = self.res(x_weight) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(x_weight[:,:,self.input_size:], (h_1, c_1))#torch.flip(x_weight[:,:,self.input_size:],dims=[2]) 
        
        output = torch.cat((output, output1), dim=2) 
        output = self.batchnorm2(output) 
        output = output+ res 
        
        hn = output[:,-1,:] 
        hn = self.sigmoid(hn) 
        
        output = torch.mean((output[:,:,:]), dim=1)
        output = self.sigmoid(output) 
        norm_output = self.batchnorm(output) 
        # temporal_weight = self.temporal_weight(norm_output) 
        temporal_weight = self.temporal_weight(norm_output) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight1(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight2(temporal_weight) 
        temporal_weight = self.selu(temporal_weight)
        
        # temporal_weight = self.temporal_weight3(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight)
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        norm_hn = self.batchnorm(hn) 
        # # hn_weight = self.hn_temporal_weight1(norm_hn) 
        hn_weight = self.hn_temporal_weight1(norm_hn) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight2(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight3(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight4(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 

        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
 
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,temporal_scores,to_q

class LSTM_t_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_t_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(4*hidden_size,1)#time_size*hidden_size
        # self.fc1 = nn.Linear(2*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight1 = nn.Linear(self.hidden_size,self.hidden_size) 
        # self.temporal_weight2 = nn.Linear(self.hidden_size,self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(self.hidden_size,self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight2 = nn.Linear(self.hidden_size,self.hidden_size) 
        # self.hn_temporal_weight3 = nn.Linear(self.hidden_size,self.hidden_size) 
        # self.hn_temporal_weight4 = nn.Linear(self.hidden_size,self.hidden_size) 
        # self.hn_temporal_weight5 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        # self.dropout = nn.Dropout(0.01)
        self.vel_weight = nn.Linear(20,2*hidden_size)
        # self.vel_weight = nn.Linear(10,2*hidden_size)
        
        # self.res1 = nn.Linear(6*2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(5)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(10)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        vel_x = torch.cat((x, future_data), dim=1) 
        result = self.vel_weight(vel_x[:,:,(self.dof+train_joint)]) 
        result = self.relu(result)
        
        x = torch.cat((x, future_data), dim=2)
        
        output, (hn, cn) = self.lstm(x[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x[:,:,self.input_size:],dims=[2])  , (h_1, c_1))#torch.flip(x_weight[:,:,self.input_size:],dims=[2]) 
        
        output = torch.cat((output, output1), dim=2) 
        output = self.batchnorm2(output) 
        
        hn = output[:,-1,:] 
        hn = self.sigmoid(hn) 
        
        output = torch.mean((output[:,:,:]), dim=1)
        output = self.sigmoid(output) 
        # norm_output = self.batchnorm(output) 
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight1(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight2(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight)
        
        # temporal_weight = self.temporal_weight3(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight)
        
        # norm_hn = self.batchnorm(hn)
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight2(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight3(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight4(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(hn_q,temporal_outputs)
        short_outputs = torch.mul(to_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,temporal_scores,to_q

class LSTM_t2_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_t2_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm

        self.fc = nn.Linear(10,1)#time_size*hidden_size
        self.fn = input_size

        self.vel_weight = nn.Linear(10,10)

        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(10)
        self.batchnorm2 = nn.BatchNorm1d(10)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定
        temp1 = x[:,:,train_joint]#(self.dof+train_joint)
        # temp2 = x[:,:,self.dof+train_joint]
        # temp = torch.cat((temp1, temp2), dim=1)
        result = self.vel_weight(temp1) 
        result = self.relu(result)
        tensor1 = x[:,:,0:self.dof]
        tensor2 = x[:,:,(self.dof):(2*self.dof)]

        x = torch.cat((tensor1, tensor2), dim=2)
               
        output, (hn, cn) = self.lstm(x[:,:,:], (h_0, c_0)) 

        output = self.batchnorm1(output)

        temporal_scores = output
        output = torch.mean(output, dim=2)
        result = nn.functional.softmax(result, dim=-1)
        output = torch.mul(output,result)
        
        output = self.relu(output) 
        out1 = self.fc(output) 
        
        return out1,temporal_scores,result



class LSTM_s_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_s_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes 
        self.num_layers = num_layers #number of layers 
        self.input_size = input_size #input size 
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2*self.input_size, out_channels=2*self.input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight1 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight2 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight3 = nn.Linear(2*self.input_size,2*self.input_size) 
        
    
        self.res = nn.Linear(2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        
        x = torch.cat((x[:,:,0*self.dof:2*self.dof], future_data[:,:,0*self.dof:2*self.dof]), dim=2)
        norm_x = self.batchnorm(x)
        spatial_attention = self.spatial_weight(norm_x) 
        spatial_attention = self.relu(spatial_attention) 
        
        # spatial_attention = self.spatial_weight1(spatial_attention) 
        # spatial_attention = self.relu(spatial_attention) 

        # spatial_attention = self.spatial_weight2(spatial_attention) 
        # spatial_attention = self.relu(spatial_attention) 
        
        # spatial_attention = self.spatial_weight3(spatial_attention) 
        # spatial_attention = self.relu(spatial_attention) 
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,x) 
        x_weight = x_weight+x
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x_weight) 
        
        # output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        # output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output = self.batchnorm2(output) 
        output1 = self.batchnorm2(output1) 
        output = torch.cat((output, output1), dim=2) 
        
        output = output+ res 
        
        hn = output[:,-1,:]
        hn = self.sigmoid(hn) 

        # outputs = self.batchnorm(hn)
        hn = self.fc1(hn)
        outputs = self.relu(hn) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores


class LSTM_st2_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st2_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes 
        self.num_layers = num_layers #number of layers 
        self.input_size = input_size #input size 
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        # self.fc1 = nn.Linear(2*hidden_size,hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=2*input_size, out_channels=2*input_size, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=1)
        # )
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2*self.input_size, out_channels=2*self.input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
        self.spatial_weight1 = nn.Linear(2*self.input_size,2*self.input_size) 
        self.spatial_weight2 = nn.Linear(2*self.input_size,2*self.input_size) 
        self.spatial_weight3 = nn.Linear(2*self.input_size,2*self.input_size) 
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        # self.spatial_weight = nn.Linear(2*self.dof,2*self.dof) 
        # self.spatial_weight1 = nn.Linear(2*self.dof,2*self.dof) 
        # self.spatial_weight2 = nn.Linear(2*self.dof,2*self.dof) 
        # self.spatial_weight3 = nn.Linear(2*self.dof,2*self.dof) 
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
        self.vel_weight1 = nn.Linear(20,2*hidden_size)
        # self.vel_weight2 = nn.Linear(seq_length,10)
        self.res = nn.Linear(2*input_size,2*hidden_size)
        # self.res = nn.Linear(2*self.dof,2*hidden_size)
        # self.res1 = nn.Linear(6*2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定

        x = torch.cat((x[:,:,0*self.dof:2*self.dof], future_data[:,:,0*self.dof:2*self.dof]), dim=2)
        
        temp1 = x[:,:,self.dof+train_joint]#(self.dof+train_joint)
        temp2 = x[:,:,3*self.dof+train_joint]
        temp = torch.cat((temp1, temp2), dim=1)
        result1 = self.vel_weight1(temp) 
        # result2 = self.vel_weight1(temp2)
        result = result1
        result = self.relu(result)
        # norm_x = self.batchnorm(x)
        spatial_attention = self.spatial_weight(x) 
        spatial_attention = self.sigmoid(spatial_attention) 
        
        spatial_attention = self.spatial_weight1(spatial_attention) 
        spatial_attention = self.sigmoid(spatial_attention) 

        spatial_attention = self.spatial_weight2(spatial_attention) 
        spatial_attention = self.sigmoid(spatial_attention) 
        
        spatial_attention = self.spatial_weight3(spatial_attention) 
        spatial_attention = self.sigmoid(spatial_attention) 
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,x) 
        x_weight = x_weight+x
        
        # x_weight = x_weight.permute(0, 2, 1) 
        # x_weight = self.conv(x_weight) 
        # x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x_weight) 
        
        # output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        # output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output = torch.cat((output, output1), dim=2) 
        output = output+ res 

        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight1(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight2(temporal_weight) 
        temporal_weight = self.selu(temporal_weight)
        
        temporal_weight = self.temporal_weight3(temporal_weight) 
        temporal_weight = self.selu(temporal_weight)
        
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        output = torch.mul(temporal_scores,output) 
        
        output = torch.mean(output, dim=2)
        result = nn.functional.softmax(result, dim=-1)
        output = torch.mul(output,result)
        to_q = result
        output = self.relu(output) 
        out1 = self.fc(output) 
        
        return out1,spatial_scores,temporal_scores,to_q

class LSTM_corrP_s_Attention_Bi(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_corrP_s_Attention_Bi, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=7, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2*input_size, out_channels=2*input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
    
        self.res = nn.Linear(2*input_size,2*hidden_size)
        # self.res1 = nn.Linear(6*2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        
        x = torch.cat((x, future_data), dim=2)
        x = x[:,:,self.dof:2*self.dof]
        # norm_x = self.batchnorm(x)
        spatial_scores = torch.zeros((128,10,18)) 
        # x_weight = norm_x.permute(0, 2, 1) 
        # x_weight = self.conv(x_weight) 
        # x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x) 
        
        output, (hn, cn) = self.lstm(x[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output = torch.cat((output, output1), dim=2) 
        output = self.batchnorm(output) 
        output = output+ res 
        
        hn = output[:,-1,:]
        hn = self.sigmoid(hn) 

        outputs = self.batchnorm(hn)
        hn = self.fc1(hn)
        outputs = self.relu(hn) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x):
        avg = self.avg_pool(x).squeeze(-1).squeeze(-1)
        channel_attention = torch.sigmoid(self.fc(avg).unsqueeze(-1).unsqueeze(-1))
        return x * channel_attention

class LSTM_st_Attention_Bi2(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_Attention_Bi2, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        # self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        #   num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(4*hidden_size,1)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight2 = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight3 = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
        
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.dropout = nn.Dropout(0.01)
        self.vel_weight = nn.Linear(10,2*hidden_size)
        self.res = nn.Linear(input_size,2*hidden_size)
        self.res1 = nn.Linear(6*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(6)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定
        
        result = self.vel_weight(x[:,:,(self.dof+train_joint)]) 
        result = self.relu(result)
        
        # norm_x = self.batchnorm(x)
        
        spatial_key = x.permute(0, 2, 1) 
        spatial_key = self.conv(spatial_key) 
        spatial_key = spatial_key.permute(0, 2, 1)
        spatial_key = self.batchnorm2(spatial_key)
        
        spatial_attention = self.spatial_weight(spatial_key)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.batchnorm2(spatial_attention)
        
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.batchnorm2(spatial_attention)

        # spatial_attention = self.spatial_weight2(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        # spatial_attention = self.batchnorm(spatial_attention)
        
        # spatial_attention = self.spatial_weight3(spatial_attention) 
        # spatial_attention = self.sigmoid(spatial_attention)
        # spatial_attention = self.batchnorm(spatial_attention)
        # spatial_attention = spatial_attention + spatial_key
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,spatial_key) 
        
        # x_weight = x_weight.permute(0, 2, 1) 
        # x_weight = self.conv(x_weight) 
        # x_weight = x_weight.permute(0, 2, 1)
        
        x_weight = x_weight.permute(0, 2, 1)
        x_weight = self.conv1(x_weight)
        x_weight = x_weight.permute(0, 2, 1)
        x_weight = self.batchnorm2(x_weight)
        
        res = self.res(x_weight) 
        res = self.relu(res)
        res1 = x_weight.reshape(x_weight.shape[0],-1)
        res1 = self.res1(res1)
        res1 = self.relu(res1)
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 

        hn = output[:,-1,:]
        hn = hn+res1
        hn = self.sigmoid(hn)
        
        output = output+ res 
        output = self.sigmoid(output) 
        
        norm_output = self.batchnorm1(output)
        temporal_weight = self.temporal_weight(norm_output) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight1(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight2(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight)
        
        # temporal_weight = self.temporal_weight3(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight)
        
        norm_hn = self.batchnorm(hn)
        hn_weight = self.hn_temporal_weight1(norm_hn) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight2(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight3(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight4(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        temporal_mean1 = torch.mean((temporal_outputs[:,:,:]), dim=1)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_mean1)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,temporal_scores,to_q

class LSTM_st_Attention_Bi4(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_Attention_Bi4, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(4*hidden_size,1)#time_size*hidden_size
        # self.fc1 = nn.Linear(2*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2*input_size, out_channels=2*input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
        self.spatial_weight1 = nn.Linear(2*self.input_size,2*self.input_size) 
        self.spatial_weight2 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight3 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
        
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight5 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        self.dropout = nn.Dropout(0.01)
        self.vel_weight = nn.Linear(20,2*hidden_size)
        self.res = nn.Linear(2*input_size,2*hidden_size)
        # self.res1 = nn.Linear(6*2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001
    
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        
        vel_x = torch.cat((x, future_data), dim=1) 
        result = self.vel_weight(vel_x[:,:,(self.dof+train_joint)]) 
        result = self.layernorm1(result)
        result = self.relu(result)
        
        x = torch.cat((x, future_data), dim=2)
        
        norm_x = self.batchnorm(x)
        spatial_attention = self.spatial_weight(norm_x)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight3(spatial_attention) 
        # spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,x) 
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x_weight) 
        res = self.relu(res)
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[1]), (h_1, c_1)) 
        
        output = torch.cat((output, output1), dim=2) 
        
        output = self.batchnorm2(output) 
        output = output+ res 
        hn = output[:,-1,:]
        hn = self.batchnorm(hn) 
        
        norm_output = self.batchnorm2(output)
        temporal_weight = self.temporal_weight(norm_output) 
        temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight1(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight2(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight)

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        temporal_mean1 = torch.mean((temporal_outputs[:,:,:]), dim=1)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_mean1)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,temporal_scores,to_q


class AbsMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AbsMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
         

    def forward(self, x):
        # 获取绝对值
        abs_x = torch.abs(x)
        
        # 计算最大值和最大值的索引
        max_vals, max_indices = torch.max(abs_x.unfold(2, self.kernel_size, self.stride), dim=2)
        
        # 使用索引从原始张量中获取对应的值
        batch_size, channels, length = x.size()
        max_vals_with_sign = torch.zeros_like(max_vals)

        for i in range(batch_size):
            for j in range(channels):
                for k in range(max_vals.size(2)):
                    # 获取绝对值最大元素的索引
                    idx = max_indices[i, j, k]
                    # 从原始张量中提取相应的值
                    max_vals_with_sign[i, j, k] = x[i, j, idx + k * self.stride]  # 注意处理步幅

        return max_vals_with_sign

class ContactComp(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(ContactComp, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm_delta = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm 
        # self.rnn = nn.RNN(input_size=2*dof, hidden_size=hidden_size, 
        #                   num_layers=num_layers, batch_first=True) #GRU 
        # self.fc = nn.Linear(1*self.hidden_size,1) 
        self.fc = nn.Linear(1*self.hidden_size,1) 
        self.fc1 = nn.Linear(self.input_size*self.seq_length,self.input_size*self.seq_length) 
        # self.fc2 = nn.Linear(1*self.dof,2*self.dof) 
        # self.fc3 = nn.Linear(1*self.dof,2*self.dof) 
        # self.fc4 = nn.Linear(1*self.dof,2*self.dof) 
        self.fn = input_size
        
        # self.laplacian_filter = torch.tensor([[1, -2, 1]], dtype=torch.float32).unsqueeze(0) 

        self.conv1 = nn.Conv1d(12, 1, 2)
        self.conv2 = nn.Conv1d(12, 1, 2)
        
        # self.conv1.weight.data = self.laplacian_filter
        # self.conv2.weight.data = self.laplacian_filter
        self.pool = nn.MaxPool1d(2)

        # self.spatial_weight = nn.Linear(2*self.dof,2*self.dof) 
        # self.spatial_weight1 = nn.Linear(8,8) 

        self.hn_temporal_weight1 = nn.Linear(1*self.hidden_size,self.hidden_size) #1*self.seq_length
        # self.hn_temporal_weight1 = nn.Linear(1*self.seq_length*self.input_size,self.hidden_size) #1*self.seq_length
        
        
        # self.batchnorm = nn.BatchNorm1d(10)
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid() 
        self.selu = nn.SELU() 
     
    def forward(self,x,train_joint,future_data,oldref):
                
        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #速度感知
        vel_x = x[:,:,:2*self.dof] 
        estimated_i = x[:,:,2*self.dof:3*self.dof] 
        vel_x = vel_x[:,:,(train_joint+self.dof)] 
        deltav = future_data[:,:,1*self.dof:2*self.dof] 
        deltaq = oldref 
        
        x = x[:,:,:2*self.dof] 
        #运动残差 
        # delta_old = torch.cat((deltaq[:,:,:],deltav[:,:,:]), dim=2)#,pos_x.unsqueeze(2),testvel.unsqueeze(2),pos_x.unsqueeze(2)
        
        q = x[:,:,:1*self.dof] 
        v = x[:,:,1*self.dof:2*self.dof] 
        delta1 = torch.cat((deltaq[:,:,:],deltav[:,:,:]), dim=2) 
        
        attentionTest = torch.cat((q[:,:,:],v[:,:,:]), dim=2) 
        
        # delta = self.sigmoid(self.fc1(delta1))+self.sigmoid(self.fc2(delta2))+self.sigmoid(self.fc3(delta3))+self.sigmoid(self.fc4(i))
        delta = delta1 
        delta = delta.permute(0, 2, 1) 
        delta = self.conv1(delta) 
        
        delta = self.relu(delta) 
        delta = self.pool(delta) 
        delta = delta.permute(0, 2, 1) 
        
        # spatial_attention = self.spatial_weight(attentionTest) 
        residualcomp = torch.cat((deltaq[:,:,:],deltav[:,:,:]), dim=2) 
        residual = self.sigmoid(self.fc1(residualcomp.reshape(-1,self.input_size*self.seq_length)))
        residual = residual.reshape(-1,self.seq_length,self.input_size) 
        attentionTest = attentionTest#+residual 
        attentionTest = attentionTest.permute(0, 2, 1) 
        spatial_attention = self.conv2(attentionTest) 
        spatial_attention = self.relu(spatial_attention) 
        spatial_attention = self.pool(spatial_attention) 
        spatial_attention = spatial_attention.permute(0, 2, 1) 
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        
        delta = torch.mul(delta,spatial_attention) 
        # delta = delta+spatial_attention 
        
        # spatial_scores = torch.zeros(delta.shape) 

        output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 

        hnd = hnd.squeeze(0) 

        temporal_weight = self.hn_temporal_weight1(hnd) 
        temporal_weight = self.selu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1) 
        hnd = torch.mul(temporal_scores,hnd) 
        # hnd = hnd.reshape(-1,self.seq_length*self.hidden_size) 
        # temporal_scores = torch.zeros(hnd.shape) 
        hnd = self.selu(hnd) 
        out1 = self.fc(hnd)#+self.selu(self.fc1(i.reshape(-1,self.seq_length*self.dof)))#+self.selu(self.fc1(attentionTest.reshape(-1,self.seq_length*self.input_size)))
        return out1,spatial_scores,temporal_scores
    
    
class ContactComp1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(ContactComp1, self).__init__() 
        self.num_classes = num_classes #number of classes 
        self.num_layers = num_layers #number of layers 
        self.input_size = input_size #input size 
        self.hidden_size = hidden_size #hidden state 
        self.seq_length = seq_length #sequence length 
        self.dof = dof
        self.lstm_delta = nn.LSTM(input_size=self.dof, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm 
        # self.rnn = nn.RNN(input_size=2*dof, hidden_size=hidden_size, 
        #                   num_layers=num_layers, batch_first=True) #GRU 

        self.fc = nn.Linear(1*self.hidden_size,1) 
        self.fc7 = nn.Linear(1*self.seq_length,1*self.hidden_size) #*self.hidden_size
        self.fn = input_size 

        self.spatial_weight = nn.Linear(1*self.dof,1*self.dof) 
        self.hn_temporal_weight3 = nn.Linear(1*self.seq_length,1*self.hidden_size) #1*self.seq_length
        
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid() 
        self.selu = nn.SELU() 
     
    def forward(self,x,train_joint,future_data,oldref):
                
        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        estimated_i = x[:,:,2*self.dof:3*self.dof] 
        estimated_i = self.sigmoid(estimated_i)
        
        i = future_data[:,:,1*self.dof:2*self.dof] 
        deltav = future_data[:,:,0*self.dof:1*self.dof] 
        deltaq = oldref 
        
        realtau = x[:,:,3*self.dof:4*self.dof] 
        x = x[:,:,:2*self.dof] 
        
        q = x[:,:,:1*self.dof] 
        v = x[:,:,1*self.dof:2*self.dof] 
        
        # delta = torch.cat((q,v), dim=2)
        # sattention = torch.cat((v,estimated_i), dim=2)
        # sattention = self.sigmoid(sattention)

        delta = estimated_i
        
        spatial_attention = self.spatial_weight(estimated_i) 
        # spatial_attention = self.spatial_weight(sattention)
        spatial_attention = self.selu(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        delta = torch.mul(delta,spatial_scores)#+ qv 
        
        output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 

        hnd = hnd.squeeze(0)+ self.fc7(self.sigmoid(estimated_i[:,:,-1])) 
        hnd = self.selu(hnd) 
        
        # temporal_weight = self.hn_temporal_weight3(torch.cat((v[:,:,train_joint],estimated_i[:,:,train_joint]),dim=1)) 
        temporal_weight = self.hn_temporal_weight3(v[:,:,train_joint]) 
        temporal_weight = self.selu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1) 
        
        hnd = torch.mul(hnd,temporal_scores)
        hnd = self.selu(hnd) 
        out1 = self.fc(hnd)
        
        return out1,spatial_scores,temporal_scores,self.selu(torch.cat((deltaq[:,-1,:],deltav[:,-1,:]), dim=1)),out1+i[:,-1,train_joint].unsqueeze(1)#torch.cat((v[:,-1,:]), dim=1)


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
        # out1 = torch.mul(momentum_power1,momentum_power)

        return out1,spatial_scores1,temporal_scores2,momentum_power1,momentum_power1#torch.cat((v[:,-1,:]), dim=1) 



class LSTM_st_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=1*input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        self.lstm_delta = nn.LSTM(input_size=input_size+1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((8,4))
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.spatial_weight = nn.Linear(1*self.input_size,1*self.input_size) 

        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,2*hidden_size)
        
        # self.otherx_weight = nn.Linear(hidden_size,8*4)
        # self.otherx_weight = nn.Linear(hidden_size,1)
        self.otherx_weight = nn.Linear(hidden_size,1*self.input_size*10)#
        self.res = nn.Linear(input_size,2*hidden_size)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        # self.batchnorm = nn.BatchNorm1d(seq_length)
        # self.dropout = nn.Dropout(0.01)
     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #速度感知
        # vel_x = torch.cat((x[:,:,:2*self.dof]), dim=1)# , future_data
        vel_x = x[:,:,:2*self.dof] 
        pos_x = vel_x[:,:,(train_joint)] 
        vel_x = vel_x[:,:,(train_joint+self.dof)] 
        deltav = x[:,:,2*self.dof:3*self.dof] 
        deltaq = oldref 
        
        x = x[:,:,:2*self.dof] 
        future_data = future_data[:,:,:2*self.dof] 
        #运动残差 
        testvel = vel_x[:,:10] 
        # pos_x = pos_x[:,:10]
        
        delta = torch.cat((deltaq[:,:,:],deltav[:,:,:],testvel.unsqueeze(2)), dim=2)#,pos_x.unsqueeze(2),testvel.unsqueeze(2),pos_x.unsqueeze(2)
        delta = self.sigmoid(delta) 
        
        output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 
        
        hnd = hnd.squeeze(0) 
        hnd = self.selu(hnd) 
        otherx = self.otherx_weight(hnd) 
        otherx = otherx.reshape(-1,10,1*self.input_size)# 
        otherx = self.selu(otherx) 
        
        result = self.vel_weight(vel_x) 
        result = self.selu(result) 
        
        all = x[:,:,:2*self.dof] 
        # all[:,:,:2*self.dof] = all[:,:,:2*self.dof] +otherx 
        # all = self.dropout(all) 
        # all = self.batchnorm(all) 
        spatial_attention = self.spatial_weight(all) #normall[:,:,:2*self.dof] 
        spatial_attention = self.sigmoid(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,all) 
        
        x_weight[:,:,:2*self.dof]= x_weight[:,:,:2*self.dof]+otherx
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
                        
        res = self.res(x_weight)
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:], (h_0, c_0)) #self.input_size
        
        # output = torch.cat((output, output1), dim=2) 
        output = self.sigmoid(output) 
        output = output + res 
        # output = self.batchnorm2(output) 
        
        hn = output[:,-1,:] 
        hn = self.sigmoid(hn)
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.relu(hn_weight) 
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1) 
        hn = torch.mul(hn_temporal_scores,hn) 
        
        output = torch.mean(output,dim=1) 
        output = self.sigmoid(output) 
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        # outputs_for = torch.cat((output,hn),dim=1) 
        temporal_scores = torch.cat((temporal_scores,hn_temporal_scores),dim=1) 
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        
        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.selu(outputs)
        out1 = self.fc(outputs)
        
        return out1,spatial_scores,temporal_scores,to_q,x_weight

class LSTM_st_noCNN_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_noCNN_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=1*input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        # self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #                   num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_delta = nn.LSTM(input_size=input_size+1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        
        self.spatial_weight = nn.Linear(1*self.input_size,1*self.input_size) 
        # self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 

        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_test = nn.Linear(2*self.hidden_size*seq_length,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,2*hidden_size)
        
        self.otherx_weight = nn.Linear(hidden_size,1*input_size*seq_length)

        self.res = nn.Linear(1*input_size,2*hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001

     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        # h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        # c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #速度感知
        # vel_x = torch.cat((x[:,:,:2*self.dof], future_data), dim=1) 
        vel_x = x[:,:,(train_joint+self.dof)] 
        deltav = x[:,:,2*self.dof:3*self.dof]
        deltaq = oldref
        
        x = x[:,:,:2*self.dof] 
        future_data = future_data[:,:,:2*self.dof] 
        #运动残差 
        testvel = vel_x[:,:10]
        delta = torch.cat((deltaq[:,:,:],deltav[:,:,:],testvel.unsqueeze(2)), dim=2)#,pos_x.unsqueeze(2),testvel.unsqueeze(2),pos_x.unsqueeze(2)
        delta = self.sigmoid(delta)
        output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 
        hnd = hnd.squeeze(0) 
        hnd = self.selu(hnd) 
        
        otherx = self.otherx_weight(hnd) 
        otherx = otherx.reshape(-1,10,1*self.input_size)# 
        otherx = self.selu(otherx) 

        result = self.vel_weight(vel_x) 
        result = self.selu(result)

        all = torch.cat((x[:,:,:2*self.dof], future_data), dim=2)
        all = x[:,:,:2*self.dof]
        spatial_attention = self.spatial_weight(all) #normall[:,:,:2*self.dof]
        spatial_attention = self.sigmoid(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,all) 
        
        #无空间注意力可选项
        # spatial_scores = torch.zeros(all[:,:,:2*self.dof].shape)
        # x_weight = all+torch.zeros(all[:,:,:2*self.dof].shape)

        x_weight[:,:,:2*self.dof]= x_weight[:,:,:2*self.dof]+otherx
        
        res = self.res(x_weight) 
    
        output, (hn, cn) = self.lstm(x_weight[:,:,:], (h_0, c_0)) #self.input_size 
        # output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[1]), (h_1, c_1))#torch.flip(x_weight[:,:,self.input_size:],dims=[2]) 
        
        output = self.sigmoid(output) 
        output = output + res 
        # output = self.batchnorm(output) 
        

        hn = output[:,-1,:]
        hn = self.sigmoid(hn)
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.relu(hn_weight) 
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 
        
        output = torch.mean(output,dim=1) 
        output = self.sigmoid(output)
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        #无时间注意力机制
        # temporal_outputs = output+torch.zeros(output.shape)
        # temporal_scores = torch.zeros(output.shape)
        
        # hn = output+torch.zeros(hn.shape)
        # hn_temporal_scores = torch.zeros(hn.shape)
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        temporal_scores = torch.cat((temporal_scores,hn_temporal_scores),dim=1) 
        
        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.selu(outputs) 
        out1 = self.fc(outputs)
        
        return out1,spatial_scores,temporal_scores,to_q,x_weight


class LSTM_onlyt_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_onlyt_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=1*input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        self.lstm_delta = nn.LSTM(input_size=input_size+1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((8,4))
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # self.spatial_weight = nn.Linear(1*self.input_size,1*self.input_size) 

        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size)         
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.vel_weight = nn.Linear(10,2*hidden_size)
        
        # self.otherx_weight = nn.Linear(hidden_size,8*4)
        # self.otherx_weight = nn.Linear(hidden_size,1)
        self.otherx_weight = nn.Linear(hidden_size,1*self.input_size*10)#
        self.res = nn.Linear(input_size,2*hidden_size)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        # self.batchnorm = nn.BatchNorm1d(10)
        # self.dropout = nn.Dropout(0.01)
     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #速度感知
        # vel_x = torch.cat((x[:,:,:2*self.dof]), dim=1)# , future_data
        vel_x = x 
        # pos_x = vel_x[:,:,(train_joint)] 
        vel_x = vel_x[:,:,(train_joint+self.dof)] 
        deltav = x[:,:,2*self.dof:3*self.dof] 
        deltaq = oldref 
        
        x = x[:,:,:2*self.dof] 
        # future_data = future_data[:,:,:2*self.dof] 
        #运动残差 
        testvel = vel_x[:,:10] 
        # pos_x = pos_x[:,:10]
        
        delta = torch.cat((deltaq[:,:,:],deltav[:,:,:],testvel.unsqueeze(2)), dim=2)#,pos_x.unsqueeze(2),testvel.unsqueeze(2),pos_x.unsqueeze(2)
        delta = self.sigmoid(delta)
        output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 
        hnd = hnd.squeeze(0) 
        hnd = self.selu(hnd) 
        otherx = self.otherx_weight(hnd) 
        otherx = otherx.reshape(-1,10,1*self.input_size)# 
        otherx = self.selu(otherx) 
        
        result = self.vel_weight(vel_x) 
        result = self.selu(result) 
        
        # all[:,:,:2*self.dof] = all[:,:,:2*self.dof] +otherx
        # all = self.dropout(all)
        # spatial_attention = self.spatial_weight(future_data) #normall[:,:,:2*self.dof] 
        # spatial_attention = self.sigmoid(spatial_attention) 
        # spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        # x_weight = torch.mul(spatial_scores,all) 
        spatial_scores = torch.zeros(x.shape)
        # x_weight = x
        x= x+otherx
        
        x = x.permute(0, 2, 1) 
        x = self.conv(x) 
        x = x.permute(0, 2, 1) 
                        
        res = self.res(x) 
        
        output, (hn, cn) = self.lstm(x[:,:,:], (h_0, c_0)) #self.input_size
        
        # output = torch.cat((output, output1), dim=2) 
        output = self.sigmoid(output) 
        output = output + res 
        # output = self.batchnorm2(output) 
        
        hn = output[:,-1,:] 
        hn = self.sigmoid(hn)
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.relu(hn_weight) 
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1) 
        hn = torch.mul(hn_temporal_scores,hn) 
        
        output = torch.mean(output,dim=1) 
        output = self.sigmoid(output) 
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        # outputs_for = torch.cat((output,hn),dim=1) 
        temporal_scores = torch.cat((temporal_scores,hn_temporal_scores),dim=1) 
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        
        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.selu(outputs)
        out1 = self.fc(outputs)
        
        return out1,spatial_scores,temporal_scores,to_q,x

class LSTM_onlys_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_onlys_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=1*input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True) #lstm
        self.lstm_delta = nn.LSTM(input_size=input_size+1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((8,4))
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.spatial_weight = nn.Linear(1*self.input_size,1*self.input_size) 
        # self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 
                
        self.vel_weight = nn.Linear(10,2*hidden_size)
        
        self.otherx_weight = nn.Linear(hidden_size,1*input_size*10)

        self.res = nn.Linear(self.input_size,2*hidden_size)

        # self.batchnorm = nn.BatchNorm1d(10)
        # self.batchnorm2 = nn.BatchNorm1d(4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001

     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        # h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        # c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #速度感知
        # vel_x = torch.cat((x[:,:,:2*self.dof], future_data), dim=1) 
        vel_x = x[:,:,(train_joint+self.dof)] 
        deltav = x[:,:,2*self.dof:3*self.dof]
        deltaq = oldref
        
        x = x[:,:,:2*self.dof] 
        future_data = future_data[:,:,:2*self.dof] 
        #运动残差 
        testvel = vel_x[:,:10] 
        delta = torch.cat((deltaq[:,:,:],deltav[:,:,:],testvel.unsqueeze(2)), dim=2)#,pos_x.unsqueeze(2),testvel.unsqueeze(2),pos_x.unsqueeze(2)
        delta = self.sigmoid(delta)
        output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 
        hnd = hnd.squeeze(0) 
        hnd = self.selu(hnd) 
        otherx = self.otherx_weight(hnd) 
        otherx = otherx.reshape(-1,10,1*self.input_size)# 
        otherx = self.selu(otherx) 

        result = self.vel_weight(vel_x) 
        result = self.selu(result)

        all = torch.cat((x[:,:,:2*self.dof], future_data), dim=2)
        all = x[:,:,:2*self.dof]
        spatial_attention = self.spatial_weight(all) #normall[:,:,:2*self.dof]
        spatial_attention = self.sigmoid(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,all) 
        
        #无空间注意力可选项
        # spatial_scores = torch.zeros(all[:,:,:2*self.dof].shape)
        # x_weight = all+torch.zeros(all[:,:,:2*self.dof].shape)
        
        x_weight[:,:,:2*self.dof]= x_weight[:,:,:2*self.dof]+otherx
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
      
        res = self.res(x_weight) 
    
        output, (hn, cn) = self.lstm(x_weight[:,:,:], (h_0, c_0)) 
        
        # output = torch.cat((output, output1), dim=2) 
        output = self.sigmoid(output)
        output = output+res 

        hn = output[:,-1,:]

        temporal_scores = torch.zeros(hn.shape)

        to_q = result
        outputs = self.selu(hn) 
        out1 = self.fc(outputs)
        
        return out1,spatial_scores,temporal_scores,to_q,x_weight

# class LSTM_onlyt_Attention_Bi3(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
#         super(LSTM_onlyt_Attention_Bi3, self).__init__()
#         self.num_classes = num_classes #number of classes
#         self.num_layers = num_layers #number of layers
#         self.input_size = input_size #input size
#         self.hidden_size = hidden_size #hidden state
#         self.seq_length = seq_length #sequence length
#         self.dof = dof
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                           num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
#         self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                           num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
#         self.lstm_delta = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                           num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
#         self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
#         self.fn = input_size
        
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=2*input_size, out_channels=2*input_size, kernel_size=3),
#             nn.Sigmoid(),
#             nn.MaxPool1d(kernel_size=3, stride=1)
#         )
        
#         self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
#         # self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 
                
#         self.vel_weight = nn.Linear(20,2*hidden_size)
        
#         self.otherx_weight = nn.Linear(hidden_size,2*input_size*6)

#         self.res = nn.Linear(2*input_size,2*hidden_size)

#         self.batchnorm = nn.BatchNorm1d(10)
#         self.batchnorm1 = nn.BatchNorm1d(12)
#         self.batchnorm2 = nn.BatchNorm1d(6)
#         self.layernorm = nn.LazyBatchNorm1d(6)
#         self.layernorm1= nn.LazyBatchNorm1d(20)
#         self.layernorm2 = nn.LazyBatchNorm1d(10)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()
#         self.elu = nn.ELU()
#         self.prelu = nn.PReLU()
#         self.selu = nn.SELU()
#         # self.l1_lambda = 0.0001

     
#     def forward(self,x,train_joint,future_data,oldref):
#         h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
#         c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
#         h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
#         c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

#         h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
#         c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
#         #速度感知
#         vel_x = torch.cat((x[:,:,:2*self.dof], future_data), dim=1) 
#         vel_x = vel_x[:,:,(train_joint+self.dof)] 
#         deltav = x[:,:,2*self.dof:3*self.dof]
#         deltaq = oldref
        
#         x = x[:,:,:2*self.dof] 
#         future_data = future_data[:,:,:2*self.dof] 
#         #运动残差 
        
#         delta = torch.cat((deltaq[:,:,:],deltav[:,:,:]), dim=2)
#         delta = self.batchnorm(delta)
#         output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 
#         delta = self.selu(hnd)
#         otherx = self.otherx_weight(delta)
#         otherx = otherx.reshape(-1,6,2*self.input_size)

#         result = self.vel_weight(vel_x) 
#         result = self.relu(result)

#         all = torch.cat((x[:,:,:2*self.dof], future_data), dim=2)
#         normall = self.batchnorm(all)
#         spatial_attention = self.spatial_weight(normall) #normall[:,:,:2*self.dof]
#         spatial_attention = self.sigmoid(spatial_attention) 
#         spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
#         x_weight = torch.mul(spatial_scores,all) 
        
#         # norm_futuredata = self.batchnorm(future_data)
#         # # spatial_attention1 = self.spatial_weight1(normall[:,:,2*self.dof:4*self.dof]) 
#         # spatial_attention1 = self.spatial_weight1(norm_futuredata) 
#         # spatial_attention1 = self.sigmoid(spatial_attention1) 
#         # spatial_scores1 = nn.functional.softmax(spatial_attention1, dim=-1) 
#         # futuredata_weight = torch.mul(spatial_scores1,future_data) 
        
#         # spatial_scores = torch.cat((spatial_scores, spatial_scores1), dim=2) 
#         # x_weight = torch.cat((x_weight, futuredata_weight), dim=2) 

#         x_weight = x_weight.permute(0, 2, 1) 
#         x_weight = self.conv(x_weight) 
#         x_weight = x_weight.permute(0, 2, 1) 
      
#         x_weight = x_weight-otherx
        

#         res = self.res(x_weight) 
    
#         output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
#         output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[1]), (h_1, c_1))#torch.flip(x_weight[:,:,self.input_size:],dims=[2]) 
        
#         output = torch.cat((output, output1), dim=2) 
#         output = output+res 
#         output = self.batchnorm2(output) 

#         hn = output[:,-1,:]

#         temporal_scores = torch.zeros(hn.shape)

#         to_q = result
#         outputs = self.relu(hn) 
#         out1 = self.fc(outputs)
        
#         return out1,spatial_scores,temporal_scores,to_q,x_weight

class LSTM_st_Attention_no_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_Attention_no_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_delta = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1*input_size, out_channels=1*input_size, kernel_size=5),
            nn.Sigmoid(),
            # nn.AdaptiveAvgPool2d((4,8))
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 

        self.temporal_weight = nn.Linear(self.hidden_size,self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(self.hidden_size,self.hidden_size) 
                
        self.vel_weight = nn.Linear(10,hidden_size)
        
        self.otherx_weight = nn.Linear(hidden_size,input_size*10)

        self.res = nn.Linear(12,hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #速度感知
        vel_x = x[:,:,:2*self.dof] 
        vel_x = vel_x[:,:,(train_joint+self.dof)] 
        deltav = x[:,:,2*self.dof:3*self.dof]
        deltaq = oldref
        
        x = x[:,:,:2*self.dof] 
        #运动残差 
        
        delta = torch.cat((deltaq[:,:,:],deltav[:,:,:]), dim=2)#,pos_x.unsqueeze(2),testvel.unsqueeze(2),pos_x.unsqueeze(2)
        delta = self.sigmoid(delta)
        output_d, (hnd, cnd) = self.lstm_delta(delta, (h_2, c_2)) 
        hnd = hnd.squeeze(0) 
        hnd = self.selu(hnd) 
        otherx = self.otherx_weight(hnd) 
        otherx = otherx.reshape(-1,10,self.input_size)# 
        otherx = self.selu(otherx) 
        
        result = self.vel_weight(vel_x) 
        result = self.selu(result)

        old_x = x[:,:,:2*self.dof]
        
        spatial_attention = self.spatial_weight(old_x) 
        spatial_attention = self.sigmoid(spatial_attention) 
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,old_x) 
        
        x_weight = x_weight+otherx
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x_weight) 
    
        output, (hn, cn) = self.lstm(x_weight[:,:,:12], (h_0, c_0)) 
        
        output = self.sigmoid(output) 
        output = output + res 
        
        hn = output[:,-1,:]
        hn = self.sigmoid(hn)
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.relu(hn_weight) 
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 

        output = torch.mean(output,dim=1)
        output = self.sigmoid(output)
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        result = nn.functional.softmax(result, dim=-1).to(device)
        danweiT = torch.ones(output.shape[0],self.hidden_size).to(device)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 

        outputs = self.selu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.selu(outputs) 
        out1 = self.fc(outputs)
        
        return out1,spatial_scores,temporal_scores,to_q,x_weight

class LSTM_st_div_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_st_div_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        # self.lstm_delatv = nn.LSTM(input_size=2*dof, hidden_size=hidden_size,
        #                   num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        # self.lstm_delatq = nn.LSTM(input_size=2*dof, hidden_size=hidden_size,
        #                   num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        # self.spatial_weight = nn.Linear(2*self.input_size-dof,2*self.input_size-dof) 
        # self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight1 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight2 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight3 = nn.Linear(2*self.input_size,2*self.input_size) 
        
        self.spatial_weight = nn.Linear(self.dof,self.dof) 
        self.spatial_weight1 = nn.Linear(self.dof,self.dof) 
        self.spatial_weight2 = nn.Linear(self.dof,self.dof) 
        self.spatial_weight3 = nn.Linear(self.dof,self.dof) 
        # self.spatial_weight3 = nn.Linear(2*self.input_size-dof,2*self.input_size-dof) 
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
        
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight5 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        self.dropout = nn.Dropout(0.01)
        self.vel_weight = nn.Linear(20,2*hidden_size)
        self.deltavel_weight = nn.Linear(20,4*hidden_size)
        # self.deltaq_weight = nn.Linear(50,4*hidden_size)
        # self.res = nn.Linear(2*input_size-dof,2*hidden_size)
        self.res = nn.Linear(224,8*2*hidden_size)
        # self.res1 = nn.Linear(self.dof,2*hidden_size)
        # self.res1 = nn.Linear(6*(2*input_size-dof),2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        # h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        # c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        # h_3 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        # c_3 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        
        vel_x = torch.cat((x[:,:,:2*self.dof], future_data[:,:,:2*self.dof]), dim=1) 
        
        velocity = vel_x[:,:,self.dof:2*self.dof]
        position = vel_x[:,:,0:self.dof]
        
        vel_x = vel_x[:,:,(train_joint+self.dof)]
        deltav = x[:,:,2*self.dof:3*self.dof]
        deltaq = oldref
        
        delta = torch.cat((deltaq[:,:,train_joint],deltav[:,:,train_joint]), dim=1)
        # output2, (hn2, cn2) = self.lstm_delatv(delta, (h_2, c_2))
        # delta = self.batchnorm(delta)
        # output2 = output2.reshape(-1,50)
        output2 = self.deltavel_weight(delta)
        output2 = self.sigmoid(output2)
        
        result = self.vel_weight(vel_x) 
        result = self.relu(result)
        
        # x = torch.cat((x, future_data), dim=2)
        
        #分散
        
        # velocity_norm = self.batchnorm1(velocity)
        velocity_attention = self.spatial_weight(velocity) 
        velocity_attention = self.sigmoid(velocity_attention) 
        
        velocity_attention = self.spatial_weight1(velocity_attention) 
        velocity_attention = self.sigmoid(velocity_attention) 
        velocity_scores = nn.functional.softmax(velocity_attention, dim=-1) 
        velocity_weight = torch.mul(velocity_scores,velocity) 
        
        # position_norm = self.batchnorm1(position)
        position_attention = self.spatial_weight2(position) 
        position_attention = self.sigmoid(position_attention) 
        
        position_attention = self.spatial_weight3(position_attention) 
        position_attention = self.sigmoid(position_attention) 
        position_scores = nn.functional.softmax(position_attention, dim=-1) 
        position_weight = torch.mul(position_scores,position) 
        
        spatial_scores = torch.cat((position_scores, velocity_scores), dim=2) 
        x_weight = torch.cat((position_weight,velocity_weight), dim=2)
        x_weight = self.batchnorm(x_weight)
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = x_weight.reshape(-1,224) 
        res = self.res(res) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:8,:], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(x_weight[:,8:,:], (h_1, c_1))#torch.flip(x_weight[:,:,self.input_size:],dims=[2]) 
        
        output = torch.cat((output, output1), dim=2) 
        res = res.reshape(-1,8,10)
        output = output+ res 
        
        hn = output[:,-1,:]
        hn = self.sigmoid(hn) 

        output = torch.mean((output[:,:,:]), dim=1)
        output = self.sigmoid(output)
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight1(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight2(temporal_weight) 
        temporal_weight = self.selu(temporal_weight)
        
        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight2(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight3(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 
        
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        # outputs = self.batchnorm1(outputs)
        outputs = self.relu(outputs) 
        outputs = outputs+output2
        # outputs = self.batchnorm1(outputs)
        outputs = self.relu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,temporal_scores,to_q,x_weight

class LSTM_s_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_s_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_delatv = nn.LSTM(input_size=dof, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2*input_size, out_channels=2*input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        # self.spatial_weight = nn.Linear(2*self.input_size-dof,2*self.input_size-dof) 
        self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight1 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight2 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight3 = nn.Linear(2*self.input_size-dof,2*self.input_size-dof) 
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
        
        self.dropout = nn.Dropout(0.01)
        self.deltavel_weight = nn.Linear(20,2*hidden_size)
        self.res = nn.Linear(2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001

     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        
        deltav = x[:,:,2*self.dof:3*self.dof]
        deltaq = oldref
        
        x = x[:,:,:2*self.dof]
        future_data = future_data[:,:,:2*self.dof]
        
        delta = torch.cat((deltaq[:,:,train_joint],deltav[:,:,train_joint]), dim=1)
        output2 = self.deltavel_weight(delta)
        output2 = self.sigmoid(output2)
        
        x = torch.cat((x, future_data), dim=2)
        
        norm_x = self.batchnorm(x)
        spatial_attention = self.spatial_weight(norm_x)
        spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight1(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight2(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight3(spatial_attention) 
        # spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,x) 
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x_weight) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output = self.batchnorm2(output)
        output1 = self.batchnorm2(output1)
        output = torch.cat((output, output1), dim=2) 
        
        output = output+ res 
        
        hn = output[:,-1,:]
        hn = self.sigmoid(hn) 

        # outputs = self.batchnorm1(hn)
        outputs = self.relu(hn) 
        outputs = outputs+output2
        # outputs = self.batchnorm1(outputs)
        outputs = self.relu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,x_weight

class LSTM_sdV_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_sdV_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2*input_size, out_channels=2*input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        # self.spatial_weight = nn.Linear(2*self.input_size-dof,2*self.input_size-dof) 
        self.spatial_weight = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight1 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight2 = nn.Linear(2*self.input_size,2*self.input_size) 
        # self.spatial_weight3 = nn.Linear(2*self.input_size-dof,2*self.input_size-dof) 
        # self.spatial_weight4 = nn.Linear(2*self.input_size,2*self.input_size)
        
        self.dropout = nn.Dropout(0.01)
        self.deltavel_weight = nn.Linear(50,2*hidden_size)
        self.res = nn.Linear(2*input_size,2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(10)
        self.batchnorm2 = nn.BatchNorm1d(6)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001

     
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        # h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        # c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        
        # vel_x = torch.cat((x[:,:,:2*self.dof], future_data), dim=1) 
        
        #使用速度残差代替速度向量
        # deltav = x[:,:,2*self.dof:3*self.dof]
        # x[:,:,self.dof:2*self.dof] = deltav
        # x = x[:,:,:2*self.dof]
        # deltav = future_data[:,:,2*self.dof:3*self.dof]
        # future_data[:,:,self.dof:2*self.dof] = deltav
        # future_data = future_data[:,:,:2*self.dof]
        
        x = torch.cat((x, future_data), dim=2)
        
        norm_x = self.batchnorm(x)
        spatial_attention = self.spatial_weight(norm_x)
        spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight1(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight2(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight3(spatial_attention) 
        # spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1) 
        x_weight = torch.mul(spatial_scores,x) 
        
        x_weight = x_weight.permute(0, 2, 1) 
        x_weight = self.conv(x_weight) 
        x_weight = x_weight.permute(0, 2, 1) 
        
        res = self.res(x_weight) 
        
        output, (hn, cn) = self.lstm(x_weight[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x_weight[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output = torch.cat((output, output1), dim=2) 
        
        output = self.batchnorm2(output) 
        output = output+ res 
        
        hn = output[:,-1,:]
        hn = self.sigmoid(hn) 

        outputs = self.batchnorm1(hn)
        outputs = self.relu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,spatial_scores,x_weight


class LSTM_t_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_t_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_delatv = nn.LSTM(input_size=dof, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight5 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        self.dropout = nn.Dropout(0.01)
        self.vel_weight = nn.Linear(20,2*hidden_size)
        self.deltavel_weight = nn.Linear(20,4*hidden_size)
        # self.res = nn.Linear(2*input_size-dof,2*hidden_size)
        self.res = nn.Linear(2*input_size,2*hidden_size)
        # self.res1 = nn.Linear(6*(2*input_size-dof),2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(10)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001

     
    def forward(self,x,train_joint,future_data,oldref):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_2 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        #位移值根据机械臂自由度确定
        
        vel_x = torch.cat((x[:,:,:2*self.dof], future_data), dim=1) 
        # vel_x = vel_x[:,:,(self.dof+train_joint)]
        # vel_x = vel_x[:,:,(2*self.dof+train_joint)]
        vel_x = vel_x[:,:,(train_joint+self.dof)]
        deltav = x[:,:,2*self.dof:3*self.dof]
        deltaq = oldref
        
        x = x[:,:,:2*self.dof]
        future_data = future_data[:,:,:2*self.dof]
        
        delta = torch.cat((deltaq[:,:,train_joint],deltav[:,:,train_joint]), dim=1)
        output2 = self.deltavel_weight(delta)
        output2 = self.sigmoid(output2)
        
        result = self.vel_weight(vel_x) 
        result = self.relu(result)
        
        x = torch.cat((x, future_data), dim=2)
        
        output, (hn, cn) = self.lstm(x[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 

        output = self.batchnorm2(output) 
        output1 = self.batchnorm2(output1) 
        output = torch.cat((output, output1), dim=2) 
        
        
        hn = output[:,-1,:]
        hn = self.sigmoid(hn) 
        
        output = torch.mean((output[:,:,:]), dim=1)
        output = self.sigmoid(output)
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight1(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight2(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight)
        

        hn_weight = self.hn_temporal_weight1(hn) 
        hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight2(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        # hn_weight = self.hn_temporal_weight3(hn_weight) 
        # hn_weight = self.selu(hn_weight) 
        
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        # outputs = self.batchnorm1(outputs)
        outputs = self.relu(outputs) 
        outputs = outputs+output2
        # outputs = self.batchnorm1(outputs)
        outputs = self.relu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,temporal_scores,to_q

class LSTM_tdV_Attention_Bi3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_tdV_Attention_Bi3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.lstm_future = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)#time_size*hidden_size
        self.fn = input_size
        
        self.temporal_weight = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        
        self.hn_temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        self.hn_temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.hn_temporal_weight5 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        self.dropout = nn.Dropout(0.01)
        self.vel_weight = nn.Linear(20,2*hidden_size)
        self.deltavel_weight = nn.Linear(50,4*hidden_size)
        # self.res = nn.Linear(2*input_size-dof,2*hidden_size)
        self.res = nn.Linear(2*input_size,2*hidden_size)
        # self.res1 = nn.Linear(6*(2*input_size-dof),2*hidden_size)
        self.batchnorm = nn.BatchNorm1d(10)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(10)
        self.layernorm = nn.LazyBatchNorm1d(12)
        self.layernorm1= nn.LazyBatchNorm1d(10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
        # self.l1_lambda = 0.0001

     
    def forward(self,x,train_joint,future_data):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定
        
        x_train = torch.cat((x[:,:,:2*self.dof], future_data[:,:,:2*self.dof]), dim=2)
        
        deltav = x[:,:,2*self.dof:3*self.dof]
        x[:,:,self.dof:2*self.dof] = deltav
        x = x[:,:,:2*self.dof]
        deltav = future_data[:,:,2*self.dof:3*self.dof]
        future_data[:,:,self.dof:2*self.dof] = deltav
        future_data = future_data[:,:,:2*self.dof]
        
        vel_x = torch.cat((x, future_data), dim=1) 
        vel_x = vel_x[:,:,(self.dof+train_joint)]
        
        result = self.vel_weight(vel_x) 
        result = self.relu(result)
        
        output, (hn, cn) = self.lstm(x_train[:,:,:self.input_size], (h_0, c_0)) 
        output1, (hn1, cn1) = self.lstm_future(torch.flip(x_train[:,:,self.input_size:],dims=[2]), (h_1, c_1)) 
        
        output = torch.cat((output, output1), dim=2) 
        
        output = self.batchnorm2(output) 
        hn = output[:,-1,:]
        hn = self.sigmoid(hn) 
        
        output = torch.mean((output[:,:,:]), dim=1)
        output = self.sigmoid(output)
        norm_output = self.batchnorm(output)
        temporal_weight = self.temporal_weight(norm_output) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight1(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight2(temporal_weight) 
        temporal_weight = self.selu(temporal_weight)
        
        norm_hn = self.batchnorm(hn)
        hn_weight = self.hn_temporal_weight1(norm_hn) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight2(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_weight = self.hn_temporal_weight3(hn_weight) 
        hn_weight = self.selu(hn_weight) 
        
        hn_temporal_scores = nn.functional.softmax(hn_weight, dim=-1)
        hn = torch.mul(hn_temporal_scores,hn) 

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],2*self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        temporal_mean = torch.mul(to_q,temporal_outputs)
        short_outputs = torch.mul(hn_q,hn)
        
        outputs = torch.cat((temporal_mean,short_outputs),dim=1) 
        outputs = self.batchnorm1(outputs)
        outputs = self.relu(outputs) 
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs) 
        out1 = self.fc(outputs) 
        
        return out1,temporal_scores,to_q


class CNN_LSTM_st_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(CNN_LSTM_st_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fn = input_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight2 = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight3 = nn.Linear(self.input_size,self.input_size) 
        # self.spatial_weight4 = nn.Linear(self.input_size,self.input_size)
        
        
        self.temporal_weight = nn.Linear(self.hidden_size,self.hidden_size) 
        # self.temporal_weight1 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight2 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight3 = nn.Linear(2*self.hidden_size,2*self.hidden_size) 
        # self.temporal_weight4 = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        
        self.dropout = nn.Dropout(0.01)
        self.vel_weight = nn.Linear(seq_length,hidden_size)
        self.res = nn.Linear(input_size,hidden_size)
        self.layernorm = nn.LayerNorm(input_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.batchnorm = nn.BatchNorm1d(seq_length-4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
        self.l2_reg = 0.01
    
    def forward(self,x,train_joint):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定

        result = self.vel_weight(x[:,:,(self.dof+train_joint)])

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight1(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight2(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)
        
        # spatial_attention = self.spatial_weight3(spatial_attention)
        # spatial_attention = self.sigmoid(spatial_attention)

        # spatial_attention = self.spatial_weight4(spatial_attention)
        spatial_attention = self.dropout(spatial_attention)
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
        
        res = self.res(x_weight) 
        
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) 
     
        output = output + res
        output = self.batchnorm(output)
        # output = self.sigmoid(output)
        hn = output[:,-1,:]
        
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.selu(temporal_weight) 

        # temporal_weight = self.temporal_weight1(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight2(temporal_weight) 
        # temporal_weight = self.selu(temporal_weight) 
        
        # temporal_weight = self.temporal_weight3(temporal_weight)
        # temporal_weight = self.selu(temporal_weight)
         
        # temporal_weight = self.temporal_weight4(temporal_weight) 
        temporal_weight = self.dropout(temporal_weight)
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        temporal_outputs = self.batchnorm(temporal_outputs)
        temporal_mean1 = torch.mean((temporal_outputs[:,:,:]), dim=1)
        # temporal_mean2 = torch.mean((temporal_outputs[:,:,self.hidden_size:]), dim=1)
        # temporal_mean1 = temporal_outputs[:,:self.hidden_size]
        # temporal_mean2 = temporal_outputs[:,self.hidden_size:]
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        # print(temporal_mean1.shape)
        # print(hn_q.shape)
        # short_outputs = torch.mul(hn_q,temporal_mean)
        temporal_mean = torch.mul(to_q,hn)
        short_outputs = torch.mul(hn_q,temporal_mean1)
        # temporal_mean = torch.mul(to_q,temporal_mean2)
        outputs = torch.cat((temporal_mean,short_outputs),dim=1)
        outputs = self.relu(outputs)
        out1 = self.fc(outputs) 
        # 计算L2正则化项
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)  # 计算参数的L2范数
        
        return out1,spatial_scores,temporal_scores,to_q,l2_loss * self.l2_reg


class LSTM_PCA_Time_Attention_comp3(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,dof):
        super(LSTM_PCA_Time_Attention_comp3, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.dof = dof
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fn = input_size
        
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight2 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight3 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight4 = nn.Linear(self.input_size,self.input_size)
        
        
        self.temporal_weight = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight1 = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight2 = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight3 = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight4 = nn.Linear(self.hidden_size,self.hidden_size)
        
        self.vel_weight = nn.Linear(seq_length,hidden_size)
        self.res = nn.Linear(input_size,hidden_size)
        self.layernorm = nn.LayerNorm(input_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.batchnorm = nn.BatchNorm1d(seq_length)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定

        result = self.vel_weight(x[:,:,(self.dof+train_joint)])

        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight3(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)

        spatial_attention = self.spatial_weight4(spatial_attention)
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)

        res = self.res(x_weight) 
        
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) 
        
        output = self.batchnorm(output)
        hn = output[:,-1,:]
        output = self.sigmoid(output) + res
        output = self.batchnorm(output)
        # hn = output[:,self.hidden_size-1,:]
        # hn = torch.squeeze(hn)
        
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.selu(temporal_weight) 

        temporal_weight = self.temporal_weight1(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight2(temporal_weight) 
        temporal_weight = self.selu(temporal_weight) 
        
        temporal_weight = self.temporal_weight3(temporal_weight)
        temporal_weight = self.selu(temporal_weight)
         
        temporal_weight = self.temporal_weight4(temporal_weight) 
        
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        temporal_mean = torch.mean(temporal_outputs, dim=1)
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        short_outputs = torch.mul(hn_q,temporal_mean)
        temporal_mean = torch.mul(to_q,hn)
        outputs = torch.cat((temporal_mean,short_outputs),dim=1)
        outputs = self.relu(outputs)
        out1 = self.fc(outputs) 
        return out1,spatial_scores,temporal_scores,to_q


class LSTM_spatial_Time_Attention_comp_auto(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size,MLP_layers, num_layers, seq_length):
        super(LSTM_spatial_Time_Attention_comp_auto, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.MLP_layers = MLP_layers #MLP hidden layer number
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        
        self.fc = nn.Linear(2*hidden_size,1)#time_size*hidden_size
        self.fn = input_size
        
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        self.spatial_out = nn.Linear(self.input_size,self.input_size)
        
        
        # 添加隐藏层
        for i in range(MLP_layers):
            self.spatial_hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        for i in range(MLP_layers):
            self.temporal_hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.temporal_weight = nn.Linear(self.hidden_size,self.hidden_size) 

        self.temporal_out = nn.Linear(self.hidden_size,self.hidden_size)
        
        self.vel_weight = nn.Linear(seq_length,hidden_size)
        self.res = nn.Linear(input_size,hidden_size)
        self.layernorm = nn.LayerNorm(input_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.batchnorm = nn.BatchNorm1d(seq_length)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.hardswish = nn.Hardswish()
    
    def forward(self,x,train_joint):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定

        result = self.vel_weight(x[:,:,(7+train_joint)])

        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)

        for layer in self.spatial_hidden_layers:
            spatial_attention = self.sigmoid(layer(spatial_attention))
        
        spatial_attention = self.spatial_out(spatial_attention)
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)

        res = self.res(x_weight) 
        
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) 
    
        output = self.sigmoid(output) + res
        output = self.batchnorm(output)
        hn = output[:,self.hidden_size-1,:]
        
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight) 

        for layer in self.temporal_hidden_layers:
            temporal_weight = self.relu(layer(temporal_weight))
         
        temporal_weight = self.temporal_out(temporal_weight) 
        
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        temporal_mean = torch.mean(temporal_outputs, dim=1)
        result = nn.functional.softmax(result, dim=-1)
        danweiT = torch.ones(output.shape[0],self.hidden_size)
        hn_q = danweiT-result
        to_q = result
        short_outputs = torch.mul(hn_q,temporal_mean)
        temporal_mean = torch.mul(to_q,hn)
        outputs = torch.cat((temporal_mean,short_outputs),dim=1)
        outputs = self.relu(outputs)
        out1 = self.fc(outputs) 
        return out1,temporal_scores,to_q

class LSTM_PCA_Time_Attention_comp4(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_Time_Attention_comp4, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        
        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fn = input_size
        
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight2 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight3 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight4 = nn.Linear(self.input_size,self.input_size)
        
        
        self.temporal_weight = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight2 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight3 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight4 = nn.Linear(hidden_size,1)
        
        # self.vel_weight = nn.Linear(seq_length,hidden_size)
        self.res = nn.Linear(input_size,hidden_size)
        self.layernorm = nn.LayerNorm(input_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.batchnorm = nn.BatchNorm1d(seq_length)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,train_joint):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight3(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)

        spatial_attention = self.spatial_weight4(spatial_attention)
        
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
        
        res = self.res(x_weight)
        
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) 
        
        output = self.sigmoid(output) + res
        output = self.batchnorm(output)
        
        # output = output.permute(0,2,1)
        
        temporal_weight = self.temporal_weight(output) 
        temporal_weight = self.relu(temporal_weight)

        temporal_weight = self.temporal_weight1(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
        
        temporal_weight = self.temporal_weight2(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
        
        temporal_weight = self.temporal_weight3(temporal_weight)
        temporal_weight = self.relu(temporal_weight)
         
        temporal_weight = self.temporal_weight4(temporal_weight)
        
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        # temporal_outputs = torch.mul(temporal_scores,output)
        # temporal_mean = torch.mean(temporal_outputs, dim=1)
        # print(temporal_weight.shape)
        # temporal_weight = temporal_weight.permute(0,2,1)
        result1 = torch.matmul(temporal_weight, output)
        to_q = torch.zeros(128,self.hidden_size)

        outputs = self.relu(result1.squeeze(1))
        out1 = self.fc(outputs) 
        return out1,temporal_scores,to_q

class LSTM_PCA_Time_Attention_comp2(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_Time_Attention_comp2, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        
        self.fc = nn.Linear(hidden_size,1)#time_size*hidden_size
        self.fc1 = nn.Linear(2*hidden_size,hidden_size)#time_size*hidden_size
        self.fn = input_size
        self.spatial_weight = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight1 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight2 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight3 = nn.Linear(self.input_size,self.input_size) 
        self.spatial_weight4 = nn.Linear(self.input_size,self.input_size)
        
        
        # self.temporal_weight = nn.Linear(self.hidden_size*self.seq_length,self.hidden_size*self.seq_length) 
        # self.temporal_weight1 = nn.Linear(self.hidden_size*self.seq_length,self.hidden_size*self.seq_length) 
        # self.temporal_weight2 = nn.Linear(self.hidden_size*self.seq_length,self.hidden_size*self.seq_length) 
        # self.temporal_weight3 = nn.Linear(self.hidden_size*self.seq_length,self.hidden_size*self.seq_length) 
        # self.temporal_weight4 = nn.Linear(self.hidden_size*self.seq_length,self.hidden_size*self.seq_length) 
        
        self.temporal_weight = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight1 = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight2 = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight3 = nn.Linear(self.hidden_size,self.hidden_size) 
        self.temporal_weight4 = nn.Linear(self.hidden_size,self.hidden_size)

        self.res = nn.Linear(input_size,hidden_size)
        
        self.layernorm = nn.LayerNorm(input_size)
        # self.batchnorm = nn.BatchNorm1d(seq_length)
        self.batchnorm = nn.BatchNorm1d(seq_length)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
    
    def forward(self,x,train_joint):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        #位移值根据机械臂自由度确定 
        
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        spatial_attention = self.spatial_weight3(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)

        spatial_attention = self.spatial_weight4(spatial_attention)
                
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
        
        res = self.res(x_weight)

        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) 
       
        output = self.sigmoid(output) + res
        output = self.batchnorm(output)

        hn = output[:,self.seq_length-1,:]
        
        temporal_weight = self.temporal_weight(hn) 
        temporal_weight = self.selu(temporal_weight)

        temporal_weight = self.temporal_weight1(temporal_weight)
        temporal_weight = self.selu(temporal_weight)
        
        temporal_weight = self.temporal_weight2(temporal_weight)
        temporal_weight = self.selu(temporal_weight)
        
        temporal_weight = self.temporal_weight3(temporal_weight)
        temporal_weight = self.selu(temporal_weight)
         
        temporal_weight = self.temporal_weight4(temporal_weight)
        
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,hn)
        
        # temporal_mean = torch.mean(temporal_outputs, dim=1)
        outputs = self.relu(temporal_outputs)
        out1 = self.fc(outputs) 
        return out1,temporal_scores

class LSTM_ST_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_ST_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        self.fc1 = nn.Linear(hidden_size,1)#time_size*hidden_size
        
        self.temporal_weight = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size) 

        self.temporal_weight2 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight3 = nn.Linear(hidden_size,hidden_size)
        
        self.fn = 14

        self.spatial_weight = nn.Linear(self.fn,self.fn) 

        self.spatial_weight1 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight2 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight3 = nn.Linear(self.fn,self.fn)

        self.test_feature = nn.Linear(7,1)
        self.test_feature1 = nn.Linear(7,1)

        self.bp_weight = nn.Linear(self.fn*seq_length,seq_length*hidden_size)
        self.bp_weight1 = nn.Linear(seq_length*hidden_size,seq_length*hidden_size)
        
        self.ResNet = nn.Linear(hidden_size,hidden_size)
        
        self.layer_norm = nn.LayerNorm(21)
        self.layer_norm1 = nn.LayerNorm(22)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.Outputlayernorm = nn.LayerNorm(hidden_size)
        self.outputResWeight = nn.Linear(21,hidden_size)
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x,pca_components):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_attention = self.spatial_weight2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_attention = self.spatial_weight3(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
           
        test = x_weight.view(-1, self.fn*self.seq_length)#time_size*hidden_size,注意修改
        bp_x = self.bp_weight(test)
        bp_x = torch.reshape(bp_x,(-1,self.seq_length,self.hidden_size))
        
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state

        temporal_weight = self.temporal_weight(output)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight1(temporal_weight)
        temporal_weight = self.relu(temporal_weight)

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        outputs = torch.mean(temporal_outputs, dim=1)#平均池化
        outputs = outputs.view(-1, self.hidden_size)#time_size*hidden_size,注意修改 
        out1 = self.fc1(outputs) #Final Output
        return out1,spatial_scores

class LSTM_ST_Res1_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_ST_Res1_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        self.fc1 = nn.Linear(hidden_size,1)#time_size*hidden_size
        
        self.temporal_weight = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size) 
        # self.temporal_weight = nn.Linear(3,10) 
        # self.temporal_weight1 = nn.Linear(14,hidden_size) 
        self.temporal_weight2 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight3 = nn.Linear(hidden_size,hidden_size)
        
        self.fn = 14

        self.spatial_weight = nn.Linear(self.fn,self.fn) 

        self.spatial_weight1 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight2 = nn.Linear(self.fn,self.fn) 
        self.spatial_weight3 = nn.Linear(self.fn,self.fn)

        self.test_feature = nn.Linear(7,1)
        self.test_feature1 = nn.Linear(7,1)

        self.bp_weight = nn.Linear(self.fn*seq_length,seq_length*hidden_size)
        self.bp_weight1 = nn.Linear(seq_length*hidden_size,seq_length*hidden_size)
        
        self.ResNet = nn.Linear(hidden_size,hidden_size)
        
        self.layer_norm = nn.LayerNorm(21)
        self.layer_norm1 = nn.LayerNorm(22)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.Outputlayernorm = nn.LayerNorm(hidden_size)
        self.outputResWeight = nn.Linear(21,hidden_size)
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x,pca_components):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置

        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_attention = self.spatial_weight2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_attention = self.spatial_weight3(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)#注意力机制激活函数
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
           
        test = x_weight.view(-1, self.fn*self.seq_length)#time_size*hidden_size,注意修改
        bp_x = self.bp_weight(test)
        bp_x = torch.reshape(bp_x,(-1,self.seq_length,self.hidden_size))
        
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state

        output = bp_x+output 
        temporal_weight = self.temporal_weight(output)
        temporal_weight = self.relu(temporal_weight)
        temporal_weight = self.temporal_weight1(temporal_weight)
        temporal_weight = self.relu(temporal_weight)

        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)

        temporal_outputs = bp_x+temporal_outputs
        # outputs = self.relu(temporal_outputs[:,self.seq_length-1,:])
        outputs = torch.mean(temporal_outputs, dim=1)#平均池化
        outputs = outputs.view(-1, self.hidden_size)#time_size*hidden_size,注意修改 
        out1 = self.fc1(outputs) #Final Output
        return out1,spatial_scores
   
class LSTM_DMD_Time_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_DMD_Time_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=0.1,bidirectional=False) #lstm
        self.fc1 = nn.Linear(hidden_size,1)#time_size*hidden_size
        
        self.temporal_weight = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size) 
        self.temporal_weight2 = nn.Linear(hidden_size,hidden_size) 
        
        self.spatial_weight = nn.Linear(1,seq_length) 
        self.spatial_weight1 = nn.Linear(6,23) 
        self.spatial_weight2 = nn.Linear(23,23) 
        # self.key_weight = nn.Linear(21,21)
        self.test_feature = nn.Linear(7,7)
        self.test_feature1 = nn.Linear(7,7)
        self.test_feature2 = nn.Linear(7,1)
        self.test_feature3 = nn.Linear(7,1)
        self.test_feature4 = nn.Linear(7,1)
        # self.bp_weight = nn.Linear(18,18)
        
        self.layer_norm = nn.LayerNorm(21)
        self.layer_norm1 = nn.LayerNorm(22)
        self.layer_norm2 = nn.LayerNorm(5)
        self.Outputlayernorm = nn.LayerNorm(hidden_size)
        self.outputResWeight = nn.Linear(21,hidden_size)
        
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x,dmd_features):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        dmd_features = dmd_features.to(device)

        #激活函数的影响是比较明显的，可能与特征形成注意力权重的方式有关。利用注意力机制实现特征增强是没有问题的，但是问题是如何获得特征？

        test_feature = self.test_feature(x[:,:,0:7])
        test_feature = self.relu(test_feature)
        test_feature = self.test_feature3(test_feature)
        test_feature = self.relu(test_feature)
        test_feature1 = self.test_feature1(x[:,:,7:14])
        test_feature1 = self.relu(test_feature1)
        test_feature1 = self.test_feature4(test_feature1)
        test_feature1 = self.relu(test_feature1) 
        
        test = torch.cat((x,test_feature),dim=2)
        test = torch.cat((test,test_feature1),dim=2)
        # test_feature2 = self.test_feature2(x[:,:,14:21])
        # test_feature2 = self.sigmoid(test_feature2)
        
        dmd_features = dmd_features.unsqueeze(-1)
        spatial_attention = self.spatial_weight(dmd_features)
        spatial_attention = spatial_attention.permute(0,2,1)
        spatial_attention = self.relu(spatial_attention)#注意力机制激活函数
        spatial_attention = self.spatial_weight1(spatial_attention)
        spatial_attention = self.relu(spatial_attention)#注意力机制激活函数
        # spatial_attention = self.spatial_weight2(spatial_attention)
        # spatial_attention = self.relu(spatial_attention)#注意力机制激活函数
        spatial_attention = self.dropout(spatial_attention)
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,test)    


        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state

        temporal_weight = self.temporal_weight(output)
        temporal_weight = self.sigmoid(temporal_weight)
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)

        outputs = torch.mean(output, dim=1)#平均池化
        outputs = self.layer_norm2(outputs+hn)#注意力机制残差连接
        outputs = self.relu(outputs)
        outputs = outputs.view(-1, self.hidden_size)#time_size*hidden_size,注意修改
        out1 = self.fc1(outputs) #Final Output
        return out1,temporal_scores

   
class LSTM_PCA_ResSpatial_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_ResSpatial_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(hidden_size,1)#time_size*hidden_size
        
        self.temporal_weight = nn.Linear(hidden_size,hidden_size)
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size)
        
        self.spatial_weight = nn.Linear(2,21)
        self.spatial_weight1 = nn.Linear(21,21) 
        # self.key_weight = nn.Linear(12,12)
        self.test_feature = nn.Linear(7,1)
        self.test_feature1 = nn.Linear(7,1)
        self.test_feature2 = nn.Linear(7,1)
        
        # self.bp_weight = nn.Linear(18,18)
        
        # self.layer_norm = nn.LayerNorm(18)
        self.layer_norm1 = nn.LayerNorm(22)
        self.layer_norm2 = nn.LayerNorm(5)
        self.Outputlayernorm = nn.LayerNorm(hidden_size)
        self.outputResWeight = nn.Linear(22,hidden_size)
        
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x,pca_components):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        pca_key_train = pca_components.to(device)
        
        test_feature1 = self.test_feature1(x[:,:,0:7])
        test_feature2 = self.test_feature2(x[:,:,7:14])
        test_feature3 = self.test_feature(x[:,:,14:21])
        
        spatial_attention = self.spatial_weight(pca_key_train)
        spatial_attention = self.relu(spatial_attention)#注意力机制激活函数
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
        
        test = torch.cat((x_weight,test_feature1),dim=2)
        test = torch.cat((test,test_feature2),dim=2)
        test = torch.cat((test,test_feature3),dim=2)
        output, (hn, cn) = self.lstm(test, (h_0, c_0)) #lstm with input, hidden, and internal state

        temporal_weight = self.temporal_weight(output)
        temporal_weight = self.sigmoid(temporal_weight)
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        # inputRes = self.outputResWeight(test)
        # inputRes = self.relu(inputRes)  
        # outputs = self.Outputlayernorm(temporal_outputs+inputRes)

        # outputs = self.relu(outputs)
        # outputs=outputs.view(-1, self.hidden_size*self.seq_length)#time_size*hidden_size,注意修改
        # out1 = self.fc1(outputs) #Final Output
        outputs = torch.mean(output, dim=1)#平均池化
        outputs = self.layer_norm2(outputs+hn)#注意力机制残差连接
        outputs = self.relu(outputs)
        outputs=outputs.view(-1, self.hidden_size)#time_size*hidden_size,注意修改
        out1 = self.fc1(outputs) #Final Output
        return out1,temporal_scores

class LSTM_PCA_ResSpatial_Attention_withR(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PCA_ResSpatial_Attention_withR, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(hidden_size,1)#time_size*hidden_size
        
        self.temporal_weight = nn.Linear(hidden_size,hidden_size)
        self.temporal_weight1 = nn.Linear(hidden_size,hidden_size)
        
        self.spatial_weight = nn.Linear(2,21)
        self.spatial_weight1 = nn.Linear(21,21) 
        # self.key_weight = nn.Linear(12,12)
        self.test_feature = nn.Linear(7,1)
        self.test_feature1 = nn.Linear(7,1)
        self.test_feature2 = nn.Linear(7,1)
        
        # self.bp_weight = nn.Linear(18,18)
        
        # self.layer_norm = nn.LayerNorm(18)
        self.layer_norm1 = nn.LayerNorm(22)
        self.layer_norm2 = nn.LayerNorm(5)
        self.Outputlayernorm = nn.LayerNorm(hidden_size)
        self.outputResWeight = nn.Linear(22,hidden_size)
        
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x,pca_components,r):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        pca_key_train = pca_components.to(device)
        
        test_feature1 = self.test_feature1(x[:,:,0:7])
        test_feature2 = self.test_feature2(x[:,:,7:14])
        test_feature3 = self.test_feature(x[:,:,14:21])
        
        spatial_attention = self.spatial_weight(pca_key_train)
        spatial_attention = self.relu(spatial_attention)#注意力机制激活函数
        spatial_scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(spatial_scores,x)
        
        test = torch.cat((x_weight,test_feature1),dim=2)
        test = torch.cat((test,test_feature2),dim=2)
        test = torch.cat((test,test_feature3),dim=2)
        test = torch.cat((test,r),dim=2)
        output, (hn, cn) = self.lstm(test, (h_0, c_0)) #lstm with input, hidden, and internal state

        temporal_weight = self.temporal_weight(output)
        temporal_weight = self.sigmoid(temporal_weight)
        temporal_scores = nn.functional.softmax(temporal_weight, dim=-1)
        temporal_outputs = torch.mul(temporal_scores,output)
        
        # inputRes = self.outputResWeight(test)
        # inputRes = self.relu(inputRes)  
        # outputs = self.Outputlayernorm(temporal_outputs+inputRes)

        # outputs = self.relu(outputs)
        # outputs=outputs.view(-1, self.hidden_size*self.seq_length)#time_size*hidden_size,注意修改
        # out1 = self.fc1(outputs) #Final Output
        outputs = torch.mean(output, dim=1)#平均池化
        outputs = self.layer_norm2(outputs+hn)#注意力机制残差连接
        outputs = self.relu(outputs)
        outputs=outputs.view(-1, self.hidden_size)#time_size*hidden_size,注意修改
        out1 = self.fc1(outputs) #Final Output
        return out1,temporal_scores


class LSTM_Time_Attention_from_Input(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Time_Attention_from_Input, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(100,1)#time_size*hidden_size
        self.temporal_weight = nn.Linear(12,10)
        self.temporal_weight1 = nn.Linear(10,10)
        self.temporal_weight2 = nn.Linear(10,10)
        self.temporal_weight3 = nn.Linear(10,10)
        self.temporal_weight4 = nn.Linear(10,10)
        self.spatial_weight = nn.Linear(12,12)
        self.spatial_weight1 = nn.Linear(12,12)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x,pca_components):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        pca_key_train = pca_components.to(device)
        temporal_attention = self.temporal_weight(x)
        temporal_attention = self.sigmoid(temporal_attention)
        temporal_attention = self.temporal_weight1(temporal_attention)
        temporal_attention = self.sigmoid(temporal_attention)
        temporal_attention = self.temporal_weight2(temporal_attention)
        temporal_attention = self.sigmoid(temporal_attention)
        temporal_attention = self.temporal_weight3(temporal_attention)
        temporal_attention = self.sigmoid(temporal_attention)
        temporal_attention = self.temporal_weight4(temporal_attention)
        temporal_attention = self.sigmoid(temporal_attention)
        scores = nn.functional.softmax(temporal_attention, dim=-1)
        spatial_attention = self.spatial_weight(x)
        spatial_attention = self.sigmoid(spatial_attention)
        spatial_attention = self.spatial_weight1(x)
        spatial_attention = self.sigmoid(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        x_weight = torch.mul(scores,x)
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn_weight = torch.mul(scores,output)
        hn_weight = self.relu(hn_weight)
        hn_weight=hn_weight.view(-1, 100)#time_size*hidden_size,注意修改
        out1 = self.fc1(hn_weight) #Final Output
        return out1,scores

class LSTM_VAE_Time_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_VAE_Time_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(300,1)#time_size*hidden_size
        self.temporal_weight = nn.Linear(3,10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.1)
    
    
    def forward(self,x,VAE_latents):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        VAE_latents = VAE_latents.to(device)
        a = torch.tensor(VAE_latents,dtype=torch.float).to(device)
        b = a.repeat(1,30,1).to(device)
        spatial_attention = self.temporal_weight(b)
        spatial_attention = self.sigmoid(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn_weight = torch.mul(scores,output)
        hn_weight = self.tanh(hn_weight)
        hn_weight = hn_weight.view(-1, 300)#time_size*hidden_size,注意修改
        out1 = self.fc1(hn_weight) #Final Output
        return out1,scores

#空间特征提取更倾向于使用VAE和PLSR两种方法。并结合self-attention的机制进行使用
class LSTM_PLSR_Spatial_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PLSR_Spatial_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(10,1)
        self.spatial_weight = nn.Linear(10,30)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    
    def forward(self,x,y):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        pls2 = PLSRegression(n_components=10)
        x_plsr = x[:,29,:]
        y_plsr = y
        pls2.fit(x_plsr.data.cpu().numpy(), y_plsr.data.cpu().numpy())
        pls_features = pls2.transform(x_plsr.data.cpu().numpy())
        a = torch.unsqueeze( torch.tensor(pls_features,dtype=torch.float).to(device),1)
        b = a.repeat(1,14,1).to(device)
        spatial_attention = self.spatial_weight(b)
        spatial_attention = self.relu(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        scores = scores.transpose(1,2)
        x_weight = torch.mul(scores,x)
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        hn = self.relu(hn)
        out1 = self.fc1(hn) #Final Output
        return out1,scores


class LSTM_PLSR_Time_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PLSR_Time_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(300,1)
        self.spatial_weight = nn.Linear(7,10)
        self.time_weight = nn.Linear(hidden_size,10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    
    def forward(self,x,plsr):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        # pls_features = np.zeros((x.shape[0],30,1))
        pls2 = PLSRegression(n_components=7)
        # for i in range(0,29):
        # x_plsr = x[:,29,:]
        # y_plsr = y
        # pls2.fit(x_plsr.data.cpu().numpy(), y_plsr.data.cpu().numpy())
        # pls_features = pls2.transform(x_plsr.data.cpu().numpy())
        # a = torch.unsqueeze( torch.tensor(pls_features,dtype=torch.float).to(device),1)
        # b = a.repeat(1,30,1).to(device)
        # print(b.shape)
        # pls_features=torch.tensor(pls_features,dtype=torch.float).to(device)
        spatial_attention = self.spatial_weight(plsr)
        spatial_attention = self.relu(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn_weight = torch.mul(scores,output)
        hn_weight = self.tanh(hn_weight)
        hn_weight=hn_weight.view(-1, 300)
        out1 = self.fc1(hn_weight) #Final Output
        return out1,scores
   
#空间特征提取更倾向于使用VAE和PLSR两种方法。并结合self-attention的机制进行使用
class LSTM_PLSR_Spatial_Time_Attention(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_PLSR_Spatial_Time_Attention, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=False) #lstm
        self.fc1 = nn.Linear(100,1)
        self.time_weight = nn.Linear(10,10)
        self.spatial_weight = nn.Linear(7,10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    
    def forward(self,x,y):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state，单向网络层配置
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state，单向网络层配置
        
        pls2 = PLSRegression(n_components=7)
        x_plsr = x[:,9,:]
        y_plsr = y
        pls2.fit(x_plsr.data.cpu().numpy(), y_plsr.data.cpu().numpy())
        pls_features = pls2.transform(x_plsr.data.cpu().numpy())
        a = torch.unsqueeze( torch.tensor(pls_features,dtype=torch.float).to(device),1)
        b = a.repeat(1,14,1).to(device)
        spatial_attention = self.spatial_weight(b)
        spatial_attention = self.sigmoid(spatial_attention)
        scores = nn.functional.softmax(spatial_attention, dim=-1)
        scores = scores.transpose(1,2)
        x_weight = torch.mul(scores,x)
        output, (hn, cn) = self.lstm(x_weight, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn_attention = self.time_weight(output)
        hn_scores = self.relu(hn_attention)
        hn_scores = nn.functional.softmax(hn_scores, dim=-1)
        hn_weight = torch.mul(hn_scores,output)
        hn_weight = self.tanh(hn_weight)
        hn_weight=hn_weight.view(-1, self.hidden_size*self.hidden_size)
        out1 = self.fc1(hn_weight) #Final Output
        return out1,scores