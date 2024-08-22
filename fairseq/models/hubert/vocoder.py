import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm 
import pdb

LRELU_SLOPE = 0.1

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class Vocoder(torch.nn.Module):
    def __init__(self, input_dimension, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, encoder_channels=512):
        super(Vocoder, self).__init__()
        #pdb.set_trace()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.input_dimension = input_dimension
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.conv_pre = weight_norm(Conv1d(self.input_dimension, self.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock
        self.encoder_channels = encoder_channels
        #self.concate_index = self.num_upsamples
        #self.concate_index = self.num_upsamples-1
        #self.concate_index = self.num_upsamples-2
        #self.concate_index = [6,2,1,0]
        #self.concate_index = [2,1,0]
        self.concate_index = [7,6,5,4,3,2]
        self.expand_index = [1,0]
        #self.concate_index = [6,2,1,0]
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            if i in self.concate_index:
                self.ups.append(weight_norm(
                    ConvTranspose1d(self.upsample_initial_channel//(2**i) + self.encoder_channels, self.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2)))
            else:
                self.ups.append(weight_norm(
                    ConvTranspose1d(self.upsample_initial_channel//(2**i) + self.encoder_channels, self.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2, output_padding=1)))
            #else:
            #    self.ups.append(weight_norm(
            #        ConvTranspose1d(self.upsample_initial_channel//(2**i), self.upsample_initial_channel//(2**(i+1)),
            #                        k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        #self.resizes = nn.ModuleList()

        for i in range(len(self.ups)):
            #if i > 1:
            #    ch = self.upsample_initial_channel//(2**(i+1))
            #else:
            #    ch = self.upsample_initial_channel//(2**(i+1)) + 512
            ch = self.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
            #if i < len(self.ups)-1:
            #    ch = self.upsample_initial_channel//(2**(i+1)) + 512
            #self.resizes.append(Conv1d(ch, self.upsample_initial_channel//(2**(i+1)),1))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, conv_rep_list=None):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            #pdb.set_trace()
            if (i in self.concate_index) or (i in self.expand_index):
                time_step_x = x.size()[2]
                time_step_c = conv_rep_list[-i-1].size()[2]
                time_step = min(time_step_x, time_step_c)
                #x = torch.cat([x, conv_rep_list[-i-1][:,:,:time_step]],axis=1)
                x = torch.cat([x[:,:,:time_step], conv_rep_list[-i-1][:,:,:time_step]],axis=1)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            #pdb.set_trace()
            #print(x.size())
            time_step = x.size()[2]
            #print(time_step) 
            #pdb.set_trace()
            #if i < len(self.ups)-1 :
            #    x = torch.cat([x, conv_rep_list[-i-1][:,:,:time_step]],axis=1) 
            #    x = self.resizes[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            #x = self.resizes[i](x)
            #pdb.set_trace()
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        #pdb.set_trace()
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class Vocoder_v2(torch.nn.Module):
    def __init__(self, input_dimension, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, encoder_channels=512, bottleneck_channels=64):
        super(Vocoder_v2, self).__init__()
        #pdb.set_trace()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.input_dimension = input_dimension
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.conv_pre = weight_norm(Conv1d(self.input_dimension, bottleneck_channels, 7, 1, padding=3))
        resblock = ResBlock
        self.encoder_channels = encoder_channels
        self.bottleneck_channels = bottleneck_channels
        #self.concate_index = self.num_upsamples
        #self.concate_index = self.num_upsamples-1
        #self.concate_index = self.num_upsamples-2
        #self.concate_index = [6,2,1,0]
        #self.concate_index = [2,1,0]
        self.concate_index = [7,6,5,4,3,2]
        self.expand_index = [1,0]
        #self.concate_index = [6,2,1,0]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.downs.append(weight_norm(Conv1d(encoder_channels, bottleneck_channels, 1, 1)))
            if i in self.concate_index:
                self.ups.append(weight_norm(
                    ConvTranspose1d(self.bottleneck_channels, self.bottleneck_channels,
                                    k, u, padding=(k-u)//2)))
            else:
                self.ups.append(weight_norm(
                    ConvTranspose1d(self.bottleneck_channels, self.bottleneck_channels,
                                    k, u, padding=(k-u)//2, output_padding=1)))
            #else:
            #    self.ups.append(weight_norm(
            #        ConvTranspose1d(self.upsample_initial_channel//(2**i), self.upsample_initial_channel//(2**(i+1)),
            #                        k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        #self.resizes = nn.ModuleList()

        for i in range(len(self.ups)):
            #if i > 1:
            #    ch = self.upsample_initial_channel//(2**(i+1))
            #else:
            #    ch = self.upsample_initial_channel//(2**(i+1)) + 512
            ch = self.bottleneck_channels
            for j, (k, d) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
            #if i < len(self.ups)-1:
            #    ch = self.upsample_initial_channel//(2**(i+1)) + 512
            #self.resizes.append(Conv1d(ch, self.upsample_initial_channel//(2**(i+1)),1))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, conv_rep_list=None):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            #pdb.set_trace()
            if (i in self.concate_index) or (i in self.expand_index):
                x = F.leaky_relu(x, LRELU_SLOPE)
                conv_rep = self.downs[-i-1](conv_rep_list[-i-1])
                time_step_x = x.size()[2]
                time_step_c = conv_rep.size()[2]
                time_step = min(time_step_x, time_step_c)
                
                x = x[:,:,:time_step] * conv_rep[:,:,:time_step]                
                #x = torch.cat([x, conv_rep_list[-i-1][:,:,:time_step]],axis=1)
                #x = torch.cat([x[:,:,:time_step], conv_rep_list[-i-1][:,:,:time_step]],axis=1)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            #pdb.set_trace()
            #print(x.size())
            time_step = x.size()[2]
            #print(time_step) 
            #pdb.set_trace()
            #if i < len(self.ups)-1 :
            #    x = torch.cat([x, conv_rep_list[-i-1][:,:,:time_step]],axis=1) 
            #    x = self.resizes[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            #x = self.resizes[i](x)
            #pdb.set_trace()
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        #x = torch.tanh(x)
        #pdb.set_trace()
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

