import copy
import torch
import heapq
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import ModADDSTCN


def preparedata(dframe, target, validation=False, id=None):
    df_data = dframe
    df_y = df_data.copy(deep=True)[target]

    if validation:
        df_data=dframe.drop(dframe.columns[[3*id, 3*id+1, 3*id+2]], axis=1)
        
    df_x = df_data.copy(deep=True).shift(periods=1, axis=0).fillna(0.)
    
    data_x = df_x.values.astype('float32').transpose()  
    data_y = df_y.values.astype('float32').transpose()

    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    
    x, y = Variable(data_x), Variable(data_y)
    return x, y


def train(epoch, traindata, traintarget, modelname, optimizer, log_interval, epochs):
    """Trains model by performing one epoch and returns attention scores and loss."""

    modelname.train()
    x, y = traindata, traintarget
    optimizer.zero_grad()
    epochpercentage = (epoch/float(epochs))*100
    output = modelname(x)
    attentionscores = modelname.fs_attention
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()


    return attentionscores.data, loss


def findcauses(target, cuda, epochs, kernel_size, layers, log_interval, lr, optimizername, seed, dilation_c, dframe):
    """Discovers potential causes of one target time series"""
        
    torch.manual_seed(seed)
    
    X_train, Y_train = preparedata(dframe, target)
    X_train = X_train.unsqueeze(0).contiguous()
    Y_train = Y_train.unsqueeze(0).contiguous()


    input_channels = X_train.size()[1]
    targetidx = dframe.columns.get_loc(target[0])//3
    model = ModADDSTCN(input_channels, num_levels=layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)

    if cuda: 
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()


    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)
    
    scores, firstloss = train(1, X_train, Y_train, model, optimizer,log_interval,epochs)
    for ep in range(2, epochs+1):
        scores, trainloss = train(ep, X_train, Y_train, model, optimizer,log_interval,epochs)
    trainloss = trainloss.cpu().data.item()
    
    sc = scores.view(-1).cpu().detach().numpy()
    scores = np.asarray([np.average(sc[i:i+3], weights=[abs(x) for x in sc[i:i+3]]) for i in range(0, len(sc), 3)])
    s = sorted(scores, reverse=True)
    indices = np.argsort(-1 *scores)

    if len(s)<=5:
        potentials = []
        for i in indices:
            if scores[i]>.5:
                potentials.append(i)
    else:
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if s[i]<.5: #tau should be greater or equal to 1, so only consider scores >= 1
                break
            gap = s[i]-s[i+1]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)

        ind = -1
        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if index<((len(s)-1)/2): #gap should be in first half
                if index>0:
                    ind=index #gap should have index > 0, except if second score <1
                    break
                
        if ind<0:
            ind = 0
                
        potentials = indices[:ind+1].tolist()


    validated = copy.deepcopy(potentials)
    
    for idx in potentials:
        X_val, Y_val = preparedata(dframe, target, validation=True, id=idx) 
        X_val = X_val.unsqueeze(0).contiguous()
        Y_val = Y_val.unsqueeze(0).contiguous() 
        
        model_val = ModADDSTCN(input_channels-3, num_levels=layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c) 

        if cuda: 
            model_val.cuda()
            X_val = X_val.cuda()
            Y_val = Y_val.cuda()

        optimizer_val = getattr(optim, optimizername)(model_val.parameters(), lr=lr)    

        for ep in range(1, epochs+1):
            valloss = train(ep, X_val, Y_val, model_val, optimizer_val, log_interval,epochs)[-1]
        valloss = valloss.cpu().data.item()         
        
        if valloss < trainloss*0.9:
            validated.remove(idx)


   
    return validated, trainloss, scores




