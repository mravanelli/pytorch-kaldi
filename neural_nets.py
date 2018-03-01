import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import sys


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)



   
class CNN_feaproc(nn.Module):
    
    def __init__(self):
       super(CNN_feaproc,self).__init__()
       self.conv1 = nn.Conv2d(1, 100, 3)
       self.conv2 = nn.Conv2d(100, 50, 5)
       
    def forward(self, x):
       steps=x.shape[0]
       batch=x.shape[1]
       x=x.view(x.shape[0]*x.shape[1],1,-1,11)
       out = F.max_pool2d(self.conv1(x), (2, 1))
       out = F.max_pool2d(self.conv2(out), (2, 2))
       out= out.view(steps,batch,-1)
       return out
        

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class normrelu(nn.Module):

    def __init__(self):
        super(normrelu, self).__init__()


    def forward(self, x):
        dim=1
        x=F.relu(x)/(torch.max(x,dim,keepdim=True)[0])
        return x
    
#class normrelu(torch.autograd.Function):
#
#    @staticmethod
#    def forward(ctx, input):

#        ctx.save_for_backward(input)
#        return input.clamp(min=0)/(torch.max(input,1,keepdim=True)[0])

#    @staticmethod
#    def backward(ctx, grad_output):
#        """
#        # same as Relu
#        """
#        input, = ctx.saved_tensors
#        grad_input = grad_output.clone()
#        grad_input[input < 0] = 0
#        return grad_input


class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()
        
        # Reading options:
        self.input_dim=options.input_dim
        self.hidden_dim=int(options.hidden_dim)
        self.N_hid=int(options.N_hid)
        self.num_classes=options.num_classes
        self.drop_rate=float(options.drop_rate)
        self.use_batchnorm=bool(int(options.use_batchnorm))
        self.use_laynorm=bool(int(options.use_laynorm))

        self.use_cuda=bool(int(options.use_cuda))
        self.resnet=bool(int(options.resnet))
        self.skip_conn=bool(int(options.skip_conn))
        self.act=options.act
        self.cost=options.cost
        
        
        # List initialization
        self.wx  = nn.ModuleList([]) 
        self.droplay = nn.ModuleList([])
         
        if self.use_batchnorm:
         self.bn_wx  = nn.ModuleList([])
         
        if self.use_laynorm:
         self.ln  = nn.ModuleList([])
                 
        if self.act=="relu":
            self.act=nn.ReLU()
            
        if self.act=="tanh":
            self.act=nn.Tanh()
            
        if self.act=="sigmoid":
            self.act=nn.Sigmoid()
        
        if self.act=="normrelu":
            self.act=normrelu()
            
        
        curr_dim=self.input_dim
        
        for i in range(self.N_hid):
                      
          # wx initialization
          if self.use_batchnorm:
           self.wx.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
           self.bn_wx.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))
          else:
           self.wx.append(nn.Linear(curr_dim, self.hidden_dim))
           
          self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.hidden_dim,curr_dim).uniform_(-np.sqrt(0.01/(curr_dim+self.hidden_dim)),np.sqrt(0.01/(curr_dim+self.hidden_dim))))
          self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.hidden_dim)) 
          
          # layer norm initialization
          if self.use_laynorm:
             self.ln.append(LayerNorm(self.hidden_dim))
        
          # dropout
          self.droplay.append(nn.Dropout(p=self.drop_rate))
          
          curr_dim=self.hidden_dim
             
        # output layer initialization
        self.fco = nn.Linear(curr_dim, self.num_classes)
        self.fco.weight = torch.nn.Parameter(torch.zeros(self.num_classes,curr_dim))
        self.fco.bias = torch.nn.Parameter(torch.zeros(self.num_classes))

        # loss definition
        if self.cost=="nll":
         self.criterion = nn.NLLLoss()
         
        if self.cost=="mse":
         self.criterion = torch.nn.MSELoss()
               
    
    def forward(self, x,lab,test_flag):
                        
      if self.use_cuda:
          x=x.cuda()
          lab=lab.cuda()
         
      # Processing hidden layers
      for i in range(self.N_hid):

        # Feed-forward affine transformation 
        wx_out=self.wx[i](x)

        # Applying batch norm
        if self.use_batchnorm:
         wx_out=self.bn_wx[i](wx_out)         
        
        if i==0 and self.skip_conn:
          prev_pre_act= Variable(torch.zeros(wx_out.shape[0],wx_out.shape[1]))
          if self.use_cuda:
            prev_pre_act=prev_pre_act.cuda()
         
        if self.skip_conn:
          wx_out=wx_out-prev_pre_act
          if i>0:
            prev_pre_act=wx_out+prev_pre_act
        
        if self.resnet and i>1:
          wx_out=wx_out-x_prev
              
        out=self.droplay[i](self.act(wx_out))
                         
        # setup x for the next hidden layer         
        x_prev=x 
        x=out

      # computing output (done in parallel)
      out=self.fco(out)

        
      # computing loss
      if self.cost=="nll":
        pout=F.log_softmax(out,dim=1)
        pred=torch.max(pout,dim=1)[1]
        loss = self.criterion(pout, lab.long())
        err = torch.mean((pred!=lab.long()).float())
        
      if self.cost=="mse":
        loss=self.criterion(out, lab)
        pout=out
        err=Variable(torch.FloatTensor([0]))
        
      return [loss,err,pout]
  
    
class GRU(nn.Module):
    def __init__(self, options):
        super(GRU, self).__init__()
        
        # Reading options:
        self.input_dim=options.input_dim
        self.hidden_dim=int(options.hidden_dim)
        self.N_hid=int(options.N_hid)
        self.num_classes=options.num_classes
        self.drop_rate=float(options.drop_rate)
        self.use_batchnorm=bool(int(options.use_batchnorm))
        self.use_laynorm=bool(int(options.use_laynorm))

        self.use_cuda=bool(int(options.use_cuda))
        self.bidir=bool(int(options.bidir))
        self.skip_conn=bool(int(options.skip_conn))
        self.act=options.act
        self.resgate=bool(int(options.resgate))
        self.minimal_gru=bool(int(options.minimal_gru))
        self.act_gate=options.act_gate
        self.cost=options.cost
        self.twin_reg=bool(int(options.twin_reg))
        self.twin_w=float(options.twin_w)
        self.cnn_pre=bool(int(options.cnn_pre))
        
        
        # List initialization
        self.wzx  = nn.ModuleList([]) # Update Gate
        self.whx  = nn.ModuleList([]) # Candidate (feed-forward)
        
        self.uzh  = nn.ModuleList([])  # Update Gate
        self.uhh  = nn.ModuleList([])  # Candidate (recurrent)
        
        if self.resgate:
         self.wrx  = nn.ModuleList([]) 
         self.urh  = nn.ModuleList([])  
         self.bn_wrx  = nn.ModuleList([])
         
        if self.use_batchnorm:
         self.bn_wzx  = nn.ModuleList([])
         self.bn_whx  = nn.ModuleList([])
         
        if self.use_laynorm:
         self.ln  = nn.ModuleList([])
                 
        if self.act=="relu":
            self.act=nn.ReLU()
            
        if self.act=="tanh":
            self.act=nn.Tanh()
            
        if self.act=="sigmoid":
            self.act=nn.Sigmoid()
        
        if self.act=="normrelu":
            self.act=normrelu()
            
        if self.act_gate=="relu":
            self.act_gate=nn.ReLU()
            
        if self.act_gate=="tanh":
            self.act_gate=nn.Tanh()
            
        if self.act_gate=="sigmoid":
            self.act_gate=nn.Sigmoid()
        
        if self.act_gate=="normrelu":
            self.act_gate=normrelu()
        
        curr_dim=self.input_dim
        
        if self.cnn_pre: # use only for 11 input frames
         curr_dim=700
         self.cnn=CNN_feaproc()
        
        for i in range(self.N_hid):
                      
          # wx initialization
          if self.use_batchnorm:
           self.wzx.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
           self.whx.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
           if self.resgate:
               self.wrx.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
               
          else:
           self.wzx.append(nn.Linear(curr_dim, self.hidden_dim))
           self.whx.append(nn.Linear(curr_dim, self.hidden_dim))
           if self.resgate:
             self.wrx.append(nn.Linear(curr_dim, self.hidden_dim))
         
          # uh initialization
          self.uzh.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
          self.uhh.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
          if self.resgate:
            self.urh.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
   
          # batch norm initialization
          if self.use_batchnorm:
           self.bn_wzx.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))
           self.bn_whx.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))
           if self.resgate:
             self.bn_wrx.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))
          
          # layer norm initialization
          if self.use_laynorm:
             self.ln.append(LayerNorm(self.hidden_dim))
             
          if self.bidir:
           curr_dim=2*self.hidden_dim
          else:
            curr_dim=self.hidden_dim   
        
        # output layer initialization
        self.fco = nn.Linear(curr_dim, self.num_classes)

        # loss definition
        if self.cost=="nll":
         self.criterion = nn.NLLLoss()
         
        if self.cost=="mse":
         self.criterion = torch.nn.MSELoss()
               
    
    def forward(self, x,lab,test_flag):
    
      # initial state
      if self.bidir or self.twin_reg:
          h_init = Variable(torch.zeros(2*x.shape[1], self.hidden_dim))
      else:
          h_init = Variable(torch.zeros(x.shape[1],self. hidden_dim))   
           
      # Drop mask initialization             
      if test_flag==0:
         drop_mask=Variable(torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.drop_rate)))
      else:
         drop_mask=Variable(torch.FloatTensor([1-self.drop_rate]))
          
      if self.use_cuda:
          x=x.cuda()
          lab=lab.cuda()
          h_init=h_init.cuda()
          drop_mask=drop_mask.cuda()
          
      if self.twin_reg:
          reg=0
          
      if self.cnn_pre:
          x=self.cnn(x)
          
      # Processing hidden layers
      for i in range(self.N_hid):
        
        # frame concatenation for bidirectional RNNs
        if self.bidir or self.twin_reg: 
            x=torch.cat([x,flip(x,0)],1)
           
        # Feed-forward affine transformation (done in parallel)
        wzx_out=self.wzx[i](x)
        whx_out=self.whx[i](x)
        if self.resgate:
         wrx_out=self.wrx[i](x)
        
        # Applying batch norm
        if self.use_batchnorm:
         wzx_out_bn=self.bn_wzx[i](wzx_out.view(wzx_out.shape[0]*wzx_out.shape[1],wzx_out.shape[2]))
         wzx_out=wzx_out_bn.view(wzx_out.shape[0],wzx_out.shape[1],wzx_out.shape[2])
                 
         whx_out_bn=self.bn_whx[i](whx_out.view(whx_out.shape[0]*whx_out.shape[1],whx_out.shape[2]))
         whx_out=whx_out_bn.view(whx_out.shape[0],whx_out.shape[1],whx_out.shape[2])
         
         if self.resgate:         
          wrx_out_bn=self.bn_wrx[i](wrx_out.view(wrx_out.shape[0]*wrx_out.shape[1],wrx_out.shape[2]))
          wrx_out=wrx_out_bn.view(wrx_out.shape[0],wrx_out.shape[1],wrx_out.shape[2]) 
         
        
        if i==0 and self.skip_conn:
          prev_pre_act= Variable(torch.zeros(whx_out.shape[0],whx_out.shape[1],whx_out.shape[2]))
          if self.use_cuda:
            prev_pre_act=prev_pre_act.cuda()
          
        if i>0 and self.skip_conn:
          prev_pre_act=pre_act    

        # Processing time steps
        hiddens = []
        pre_act = []
        h=h_init
        
        for k in range(x.shape[0]):
          zt=self.act_gate(wzx_out[k]+self.uzh[i](h))
          
          if self.resgate: 
            if self.minimal_gru:
             at=whx_out[k]+self.uhh[i](zt*h)
            else:
             rt=self.act_gate(wrx_out[k]+self.urh[i](h))
             at=whx_out[k]+self.uhh[i](rt*h)
          else:
             at=whx_out[k]+self.uhh[i](h)
             
          if self.skip_conn:
              pre_act.append(at)
              at=at-prev_pre_act[k]
              
           
          if self.use_laynorm:
              at=self.ln[i](at)
              
          hcand=self.act(at)*drop_mask
          h=(zt*h+(1-zt)*hcand)
          
          hiddens.append(h)
          
    
        # stacking hidden states
        h=torch.stack(hiddens)
        if self.skip_conn:
         pre_act=torch.stack(pre_act)

         
        # bidirectional concatenations
        if self.bidir:
         h_f=h[:,0:int(x.shape[1]/2)]
         h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
         h=torch.cat([h_f,h_b],2)
         
        if self.twin_reg:
          if not(self.bidir):
            h_f=h[:,0:int(x.shape[1]/2)]
            h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
            h=h_f
          reg=reg+torch.mean((h_f - h_b)**2)
        
        # setup x for the next hidden layer
        x=h

      # computing output (done in parallel)
      out=self.fco(h)

        
      # computing loss
      if self.cost=="nll":
        pout=F.log_softmax(out,dim=2)
        pred=torch.max(pout,dim=2)[1]
        loss=self.criterion(pout.view(h.shape[0]*h.shape[1],-1), lab.view(-1))#+1.0*reg  
        err = torch.sum((pred!=lab).float())/(h.shape[0]*h.shape[1])
        
      if self.cost=="mse":
        loss=self.criterion(out, lab)
        pout=out
        err=Variable(torch.FloatTensor([0]))
        
      if self.twin_reg:
          loss=loss+self.twin_w*reg
        
      return [loss,err,pout]
  
    
    
    
class RNN(nn.Module):
    def __init__(self, options):
        super(RNN, self).__init__()
        
        # Reading options:
        self.input_dim=options.input_dim
        self.hidden_dim=int(options.hidden_dim)
        self.N_hid=int(options.N_hid)
        self.num_classes=options.num_classes
        self.drop_rate=float(options.drop_rate)
        self.use_batchnorm=bool(int(options.use_batchnorm))
        self.use_laynorm=bool(int(options.use_laynorm))

        self.use_cuda=bool(int(options.use_cuda))
        self.bidir=bool(int(options.bidir))
        self.skip_conn=bool(int(options.skip_conn))
        self.act=options.act
        self.act_gate=options.act_gate
        self.cost=options.cost
        self.twin_reg=bool(int(options.twin_reg))
        self.twin_w=float(options.twin_w)
        
        self.cnn_pre=bool(int(options.cnn_pre))
        
       
        # List initialization
        self.wx  = nn.ModuleList([]) # Update Gate
        self.uh  = nn.ModuleList([]) # Candidate (feed-forward)
        
        if self.cnn_pre:
            self.cnn=CNN_feaproc()
        
         
        if self.use_batchnorm:
         self.bn_wx  = nn.ModuleList([])
         
        if self.use_laynorm:
         self.ln  = nn.ModuleList([])
                 
        if self.act=="relu":
            self.act=nn.ReLU()
            
        if self.act=="tanh":
            self.act=nn.Tanh()
            
        if self.act=="sigmoid":
            self.act=nn.Sigmoid()
        
        if self.act=="normrelu":
            self.act=normrelu()
            
        if self.act_gate=="relu":
            self.act_gate=nn.ReLU()
            
        if self.act_gate=="tanh":
            self.act_gate=nn.Tanh()
            
        if self.act_gate=="sigmoid":
            self.act_gate=nn.Sigmoid()
        
        if self.act_gate=="normrelu":
            self.act_gate=normrelu()
        
        curr_dim=self.input_dim
        
        if self.cnn_pre:
         curr_dim=700
        
        for i in range(self.N_hid):
                      
          # wx initialization
          if self.use_batchnorm:
           self.wx.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
          else:
           self.wx.append(nn.Linear(curr_dim, self.hidden_dim))
           
          # uh initialization
          self.uh.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
          
   
          # batch norm initialization
          if self.use_batchnorm:
           self.bn_wx.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))

          # layer norm initialization
          if self.use_laynorm:
             self.ln.append(LayerNorm(self.hidden_dim))
             
          if self.bidir:
           curr_dim=2*self.hidden_dim
          else:
            curr_dim=self.hidden_dim   
        
        # output layer initialization
        self.fco = nn.Linear(curr_dim, self.num_classes)

        # loss definition
        if self.cost=="nll":
         self.criterion = nn.NLLLoss()
         
        if self.cost=="mse":
         self.criterion = torch.nn.MSELoss()

    
    def forward(self, x,lab,test_flag):
    
      # initial state
      if self.bidir or self.twin_reg:
          h_init = Variable(torch.zeros(2*x.shape[1], self.hidden_dim))
      else:
          h_init = Variable(torch.zeros(x.shape[1],self. hidden_dim))   
           
      # Drop mask initialization             
      if test_flag==0:
         drop_mask=Variable(torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.drop_rate)))
      else:
         drop_mask=Variable(torch.FloatTensor([1-self.drop_rate]))
          
      if self.use_cuda:
          x=x.cuda()
          lab=lab.cuda()
          h_init=h_init.cuda()
          drop_mask=drop_mask.cuda()
      
      if self.twin_reg:
          reg=0
      
       # cnn pre-processing
      if self.cnn_pre:
          x=self.cnn(x)




          
      # Processing hidden layers
      for i in range(self.N_hid):
            
        # frame concatenation for bidirectional RNNs
        if self.bidir or self.twin_reg: 
            x=torch.cat([x,flip(x,0)],1)
        
        # Feed-forward affine transformation (done in parallel)
        wx_out=self.wx[i](x)

        
        # Applying batch norm
        if self.use_batchnorm:
         wx_out_bn=self.bn_wx[i](wx_out.view(wx_out.shape[0]*wx_out.shape[1],wx_out.shape[2]))
         wx_out=wx_out_bn.view(wx_out.shape[0],wx_out.shape[1],wx_out.shape[2])
         
        
        if i==0 and self.skip_conn:
          prev_pre_act= Variable(torch.zeros(wx_out.shape[0],wx_out.shape[1],wx_out.shape[2]))
          if self.use_cuda:
            prev_pre_act=prev_pre_act.cuda()
          
        if i>0 and self.skip_conn:
          prev_pre_act=pre_act    

        # Processing time steps
        hiddens = []
        pre_act = []
        h=h_init
        
        for k in range(x.shape[0]):
          at=wx_out[k]+self.uh[i](h)
             
          if self.skip_conn:
              pre_act.append(at)
              at=at-prev_pre_act[k]
               
           
          if self.use_laynorm:
              at=self.ln[i](at)
              
          h=self.act(at)*drop_mask
          
          hiddens.append(h)
          
    
        # stacking hidden states
        h=torch.stack(hiddens)
        if self.skip_conn:
         pre_act=torch.stack(pre_act)

         
        # bidirectional concatenations
        if self.bidir:
         h_f=h[:,0:int(x.shape[1]/2)]
         h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
         h=torch.cat([h_f,h_b],2)
        
        if self.twin_reg:
          if not(self.bidir):
            h_f=h[:,0:int(x.shape[1]/2)]
            h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
            h=h_f
          reg=reg+torch.mean((h_f - h_b)**2)
          

        # setup x for the next hidden layer
        x=h

      # computing output (done in parallel)
      out=self.fco(h)

        
      # computing loss
      if self.cost=="nll":
        pout=F.log_softmax(out,dim=2)
        pred=torch.max(pout,dim=2)[1]
        loss=self.criterion(pout.view(h.shape[0]*h.shape[1],-1), lab.view(-1))
        err = torch.sum((pred!=lab).float())/(h.shape[0]*h.shape[1])
        
      if self.cost=="mse":
        loss=self.criterion(out, lab)
        pout=out
        err=Variable(torch.FloatTensor([0]))
        
      if self.twin_reg:
          loss=loss+self.twin_w*reg
        
      return [loss,err,pout]
  
    
    
class LSTM(nn.Module):
    def __init__(self, options):
        super(LSTM, self).__init__()
        
        # Reading options:
        self.input_dim=options.input_dim
        self.hidden_dim=int(options.hidden_dim)
        self.N_hid=int(options.N_hid)
        self.num_classes=options.num_classes
        self.drop_rate=float(options.drop_rate)
        self.use_batchnorm=bool(int(options.use_batchnorm))
        self.use_laynorm=bool(int(options.use_laynorm))

        self.use_cuda=bool(int(options.use_cuda))
        self.bidir=bool(int(options.bidir))
        self.skip_conn=bool(int(options.skip_conn))
        self.act=options.act
        self.act_gate=options.act_gate
        self.cost=options.cost
        self.twin_reg=bool(int(options.twin_reg))
        self.twin_w=float(options.twin_w)
        self.cnn_pre=bool(int(options.cnn_pre))
        
        
        # List initialization
        self.wfx  = nn.ModuleList([]) # Forget
        self.ufh  = nn.ModuleList([]) # Forget
        
        self.wix  = nn.ModuleList([]) # Input
        self.uih  = nn.ModuleList([]) # Input  
        
        self.wox  = nn.ModuleList([]) # Output
        self.uoh  = nn.ModuleList([]) # Output  
        
        self.wcx  = nn.ModuleList([]) # Cell state
        self.uch  = nn.ModuleList([]) # Cell state  
        
        if self.use_batchnorm:
         self.bn_wfx  = nn.ModuleList([])
         self.bn_wix  = nn.ModuleList([])
         self.bn_wox  = nn.ModuleList([])
         self.bn_wcx  = nn.ModuleList([])
         
        if self.use_laynorm:
         self.ln  = nn.ModuleList([])
                 
        if self.act=="relu":
            self.act=nn.ReLU()
            
        if self.act=="tanh":
            self.act=nn.Tanh()
            
        if self.act=="sigmoid":
            self.act=nn.Sigmoid()
        
        if self.act=="normrelu":
            self.act=normrelu()
            
        if self.act_gate=="relu":
            self.act_gate=nn.ReLU()
            
        if self.act_gate=="tanh":
            self.act_gate=nn.Tanh()
            
        if self.act_gate=="sigmoid":
            self.act_gate=nn.Sigmoid()
        
        if self.act_gate=="normrelu":
            self.act_gate=normrelu()
        
        curr_dim=self.input_dim
        
        if self.cnn_pre: # use only for 11 input frames
         curr_dim=700
         self.cnn=CNN_feaproc()
        
        for i in range(self.N_hid):
                      
          # wx initialization
          if self.use_batchnorm:
           self.wfx.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
           self.wix.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
           self.wox.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
           self.wcx.append(nn.Linear(curr_dim, self.hidden_dim,bias=False))
          else:
           self.wfx.append(nn.Linear(curr_dim, self.hidden_dim))
           self.wix.append(nn.Linear(curr_dim, self.hidden_dim))
           self.wox.append(nn.Linear(curr_dim, self.hidden_dim))
           self.wcx.append(nn.Linear(curr_dim, self.hidden_dim))
           
          # uh initialization
          self.ufh.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
          self.uih.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
          self.uoh.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
          self.uch.append(nn.Linear(self.hidden_dim, self.hidden_dim,bias=False))
   
          # batch norm initialization
          if self.use_batchnorm:
           self.bn_wfx.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))
           self.bn_wix.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))
           self.bn_wox.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))
           self.bn_wcx.append(nn.BatchNorm1d(self.hidden_dim,momentum=0.05))

          # layer norm initialization
          if self.use_laynorm:
             self.ln.append(LayerNorm(self.hidden_dim))
             
          if self.bidir:
           curr_dim=2*self.hidden_dim
          else:
            curr_dim=self.hidden_dim   
        
        # output layer initialization
        self.fco = nn.Linear(curr_dim, self.num_classes)

        # loss definition
        if self.cost=="nll":
         self.criterion = nn.NLLLoss()
         
        if self.cost=="mse":
         self.criterion = torch.nn.MSELoss()
               
    
    def forward(self, x,lab,test_flag):
    
      # initial state
      if self.bidir or self.twin_reg:
          h_init = Variable(torch.zeros(2*x.shape[1], self.hidden_dim))
      else:
          h_init = Variable(torch.zeros(x.shape[1],self. hidden_dim))   
           
      # Drop mask initialization             
      if test_flag==0:
         drop_mask=Variable(torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.drop_rate)))
      else:
         drop_mask=Variable(torch.FloatTensor([1-self.drop_rate]))
          
      if self.use_cuda:
          x=x.cuda()
          lab=lab.cuda()
          h_init=h_init.cuda()
          drop_mask=drop_mask.cuda()
          
      if self.twin_reg:
          reg=0
          
      if self.cnn_pre:
          x=self.cnn(x)
          
      # Processing hidden layers
      for i in range(self.N_hid):
        
        # frame concatenation for bidirectional RNNs
        if self.bidir or self.twin_reg: 
            x=torch.cat([x,flip(x,0)],1)
           
        # Feed-forward affine transformation (done in parallel)
        wfx_out=self.wfx[i](x)
        wix_out=self.wix[i](x)
        wox_out=self.wox[i](x)
        wcx_out=self.wcx[i](x)

        
        # Applying batch norm
        if self.use_batchnorm:
         wfx_out_bn=self.bn_wfx[i](wfx_out.view(wfx_out.shape[0]*wfx_out.shape[1],wfx_out.shape[2]))
         wfx_out=wfx_out_bn.view(wfx_out.shape[0],wfx_out.shape[1],wfx_out.shape[2])
         
         wix_out_bn=self.bn_wix[i](wix_out.view(wix_out.shape[0]*wix_out.shape[1],wix_out.shape[2]))
         wix_out=wix_out_bn.view(wix_out.shape[0],wix_out.shape[1],wix_out.shape[2])
   
         wox_out_bn=self.bn_wox[i](wox_out.view(wox_out.shape[0]*wox_out.shape[1],wox_out.shape[2]))
         wox_out=wox_out_bn.view(wox_out.shape[0],wox_out.shape[1],wox_out.shape[2])

         wcx_out_bn=self.bn_wcx[i](wcx_out.view(wcx_out.shape[0]*wcx_out.shape[1],wcx_out.shape[2]))
         wcx_out=wcx_out_bn.view(wcx_out.shape[0],wcx_out.shape[1],wcx_out.shape[2])           
        
        if i==0 and self.skip_conn:
          prev_pre_act= Variable(torch.zeros(wfx_out.shape[0],wfx_out.shape[1],wfx_out.shape[2]))
          if self.use_cuda:
            prev_pre_act=prev_pre_act.cuda()
          
        if i>0 and self.skip_conn:
          prev_pre_act=pre_act    

        # Processing time steps
        hiddens = []
        pre_act = []
        c=h_init
        h=h_init
        
        for k in range(x.shape[0]):
          
          ft=self.act_gate(wfx_out[k]+self.ufh[i](h))
          it=self.act_gate(wix_out[k]+self.uih[i](h))
          ot=self.act_gate(wox_out[k]+self.uoh[i](h))
          
          at=wcx_out[k]+self.uch[i](h)
             
          if self.skip_conn:
              pre_act.append(at)
              at=at-prev_pre_act[k]
              
             
           
          if self.use_laynorm:
              at=self.ln[i](at)
              
          c=it*self.act(at)*drop_mask+ft*c
          h=ot*self.act(c)
          
          hiddens.append(h)
          
    
        # stacking hidden states
        h=torch.stack(hiddens)
        if self.skip_conn:
         pre_act=torch.stack(pre_act)

         
        # bidirectional concatenations
        if self.bidir:
         h_f=h[:,0:int(x.shape[1]/2)]
         h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
         h=torch.cat([h_f,h_b],2)
         
        if self.twin_reg:
          if not(self.bidir):
            h_f=h[:,0:int(x.shape[1]/2)]
            h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
            h=h_f
          reg=reg+torch.mean((h_f - h_b)**2)
        
        # setup x for the next hidden layer
        x=h

      # computing output (done in parallel)
      out=self.fco(h)

        
      # computing loss
      if self.cost=="nll":
        pout=F.log_softmax(out,dim=2)
        pred=torch.max(pout,dim=2)[1]
        loss=self.criterion(pout.view(h.shape[0]*h.shape[1],-1), lab.view(-1)) 
        err = torch.sum((pred!=lab).float())/(h.shape[0]*h.shape[1])
        
      if self.cost=="mse":
        loss=self.criterion(out, lab)
        pout=out
        err=Variable(torch.FloatTensor([0]))
        
      if self.twin_reg:
          loss=loss+self.twin_w*reg
        
      return [loss,err,pout]
  
    
   
