"""
baselines.py: contains all your network structure definition
including layers definition and forward pass function definition
"""
# PyTorch and neural network imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import math
torch.cuda.init()

# set the randomness to keep reproducible results
torch.manual_seed(0)
np.random.seed(0)

def conv_out_dim_calculate(in_size, out_channels, kernel_size, stride, padding):
    out_width = math.floor((in_size + (2 * padding) - kernel_size)/stride) + 1
    return (out_width**2) * out_channels

conv_kernel_size = 5
conv_out_channels = 10
conv_stride = 1
conv_padding = 2

dummy = 1
# input size to your mlp network
mlp_input_size = 784 # TODO1
# final output size of your mlp network (output layer)
mlp_output_size = 10 # TODO2
# TODO3: you may need to experiment a bit (width of your hidden layer)
mlp_hidden_size = 100

class BaselineMLP(nn.Module):
    def __init__(self):
        """
        A multilayer perceptron model
        Consists of one hidden layer and 1 output layer (all fully connected)
        """
        super(BaselineMLP, self).__init__()
        # a fully connected layer from input layer to hidden layer
        # mlp_input_size denotes how many input neurons you have
        # mlp_hiddent_size denotes how many hidden neurons you have
        self.fc1 = nn.Linear(mlp_input_size, mlp_hidden_size).cuda()
        # a fully connected layer from hidden layer to output layer
        # mlp_output_size denotes how many output neurons you have
        self.fc2 = nn.Linear(mlp_hidden_size, mlp_output_size).cuda()
    
    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying 
        logistic activation function after hidden layer.
        """
        # pass X from input layer to hidden layer
        out = self.fc1(X.cuda()).cuda()
        # apply an activation function to the output of hidden layer
        out = torch.sigmoid(out).cuda()
        # pass output from hidden layer to output layer
        out = self.fc2(out).cuda()
        # return the feed forward output
        # you don't need to apply another activation function here if
        # the loss function you use already implement it for you
        return out


class BaselineCNN(nn.Module):
    def __init__(self, in_dim, in_channels, n_classes):
        """
        A basic convolutional neural network model for baseline comparison.
        Consists of one Conv2d layer, followed by 1 fully-connected (FC) layer:
        conv1 -> fc1 (outputs)
        """
        super(BaselineCNN, self).__init__()
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.conv_out_dim = conv_out_dim_calculate(self.in_dim, conv_out_channels, conv_kernel_size, conv_stride, conv_padding)

        self.n_classes = n_classes

        self.conv = nn.Conv2d(self.in_channels, conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding).cuda()
        self.fc1 = nn.Linear(self.conv_out_dim, n_classes).cuda()
        self.softmax = nn.Softmax().cuda()
        self.tanh = nn.Tanh().cuda()
        # TODO7: define different layers


    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying 
        non-linearities after each layer.
        
        Note that this function *needs* to be called "forward" for PyTorch to 
        automagically perform the forward pass.

        You may need the function "num_fc_features" below to help implement 
        this function
        
        Parameters: X --- an input batch of images
        Returns:    out --- the output of the network
        """
        # TODO8: define the forward function
        out = self.tanh(self.conv(X.cuda()).view(-1, self.conv_out_dim)).cuda()
        out = self.fc1(out).cuda()
        return out

    """
    Count the number of flattened features to be passed to fully connected layers
    Parameters: inputs --- 4-dimensional [batch x num_channels x conv width x conv height]
                            output from the last conv layer
    Return: num_features --- total number of flattened features for the last layer
    """
    def num_fc_features(self, inputs):
        
        # Get the dimensions of the layers excluding the batch number
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

"""
TODO: you may need to define your new neural network here
"""


class TheNameOfYourClass(nn.Module):
    def __init__(self, in_dim, in_channels, n_classes):
        """
        A basic convolutional neural network model for baseline comparison.
        Consists of one Conv2d layer, followed by 1 fully-connected (FC) layer:
        conv1 -> fc1 (outputs)
        """
        super(TheNameOfYourClass, self).__init__()
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.conv_out_dim1 = conv_out_dim_calculate(in_size=self.in_dim, out_channels=10,
                                                    kernel_size=1, stride=1, padding=0)
        self.conv_out_dim2 = conv_out_dim_calculate(in_size=self.conv_out_dim1, out_channels=25,
                                                    kernel_size=5, stride=1, padding=2)
        # self.conv_out_dim = conv_out_dim_calculate(self.in_dim, conv_out_channels, conv_kernel_size, conv_stride,
        #                                            conv_padding)

        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=10, kernel_size=1,
                               stride=1, padding=0).cuda()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=25, kernel_size=5,
                               stride=1, padding=2).cuda()
        # self.conv = nn.Conv2d(self.in_channels, conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride,
        #                       padding=conv_padding).cuda()
        self.pl1 = nn.MaxPool2d(2,2).cuda()
        self.dropout = nn.Dropout2d(.5).cuda()
        self.fc1 = nn.Linear(4900, 2500).cuda()
        self.fc2 = nn.Linear(2500, self.n_classes).cuda()
        self.softmax = nn.Softmax().cuda()
        self.relu = nn.ReLU().cuda()
        self.tanh = nn.Tanh().cuda()
        # TODO7: define different layers

    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.

        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.

        You may need the function "num_fc_features" below to help implement
        this function

        Parameters: X --- an input batch of images
        Returns:    out --- the output of the network
        """
        # TODO8: define the forward function
        out = self.relu(self.conv1(X.cuda()))
        out = self.relu(self.conv2(out))
        out = self.pl1(out)
        B,C,H,W = out.shape
        out = out.view(B, C*H*W)
        out = self.relu(self.fc1(out).cuda())
        out = self.dropout(out).cuda()
        out = self.fc2(out).cuda()
        return out

    """
    Count the number of flattened features to be passed to fully connected layers
    Parameters: inputs --- 4-dimensional [batch x num_channels x conv width x conv height]
                            output from the last conv layer
    Return: num_features --- total number of flattened features for the last layer
    """

    def num_fc_features(self, inputs):
        # Get the dimensions of the layers excluding the batch number
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features
