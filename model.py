from torch import nn, ones, autograd
from depthwise import DepthwiseNet



class ModADDSTCN(nn.Module):
    
    def __init__(self, input_size, num_levels, kernel_size, cuda, dilation_c):
        super(ModADDSTCN, self).__init__()
        
        self.input_size = input_size
        self.dwn = DepthwiseNet(input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv2d(input_size//3, 1, 1)
        self._attention = ones(input_size,1) 
        self._attention = autograd.Variable(self._attention, requires_grad=False)
        self.fs_attention = nn.Parameter(self._attention.data)
        
        if cuda:
            self.dwn = self.dwn.cuda() 
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()
                  
    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)       
        
    def forward(self, x):
        y1=self.dwn(x*nn.functional.softmax(self.fs_attention, dim=0))
        y1=y1.reshape(self.input_size//3, 3, y1.shape[-1])
        y1 = self.pointwise(y1)
        return y1