from __future__ import print_function, division

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable 
from torch.utils.data import Dataset
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os 

class NeuralNet(nn.Module):
	"Basic NN model for 3 layers"
	def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
		"Creates NN object for 3 layers sizing"
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size_1)
		self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.fc3 = nn.Linear(hidden_size_2, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		" Forward propogates nn"
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out 

class NeuralNet_sherpa_optimize(nn.Module):
	" nn Class that allows sherpa optmization"
	def __init__(self, input_size,output_size, Parameters):
		" Makes NN object that uses sherpa based object to unpack parameters for nn, generally used for flexiable optimization"
		super(NeuralNet_sherpa_optimize, self).__init__()
		#unpack sherpa package note 2 properties used outside
		hidden_size_1=Parameters["H_l1"]
		hidden_size_2=int(Parameters["H_l1"]/2) #found by first trial set anaylsis
		activate=Parameters["activate"]
		#general dictionary for activation functions
		self.dic_activation={
			"nn.Hardswish" : nn.Hardswish(),
			"nn.PReLU" : nn.PReLU(),
			"nn.ReLU" : nn.ReLU(),
			"nn.Sigmoid" : nn.Sigmoid(),
			"nn.LeakyReLU" :nn.LeakyReLU(),
		} 
		self.fc1 = nn.Linear(input_size, hidden_size_1)
		self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.fc3 = nn.Linear(hidden_size_2, output_size)
		# set activation
		self.activate = self.dic_activation[activate]

	def forward(self, x):
		"Forward Propogates"
		out = self.fc1(x)
		out = self.activate(out)
		out = self.fc2(out)
		out = self.activate(out)
		out = self.fc3(out)
		return out 

class MyDataset(Dataset):
	"Dataset object used to load data for use"

	def __init__(self,frame,interest,features):
		"unpacks and loads into tensors the features and objective"
		x=frame[features].values
		y=frame[interest].values

		self.x_train=torch.tensor(x,dtype=torch.float32)
		y_train=torch.tensor(y,dtype=torch.float32)
		self.y_train=y_train.view(-1,1)

	def __len__(self):
		return len(self.y_train)

	def __getitem__(self,idx):
		return self.x_train[idx],self.y_train[idx]

class NeuralNet_sherpa_optimize_1(nn.Module):
	"nn used to optmize on device not found in cpu, method was depreciated"
	def __init__(self, input_size,output_size, Parameters):
		super(NeuralNet_sherpa_optimize_1, self).__init__()
		#unpack sherpa package note 2 properties used outside
		hidden_size_1=Parameters["H_l1"]
		hidden_size_2=int(Parameters["H_l1"]/2) #found by first trial set anaylsis
		activate=Parameters["activate"]
		#general dictionary for activation functions
		self.dic_activation={
			"nn.Hardswish" : nn.Hardswish(),
			"nn.PReLU" : nn.PReLU(),
			"nn.ReLU" : nn.ReLU(),
			"nn.Sigmoid" : nn.Sigmoid(),
			"nn.LeakyReLU" :nn.LeakyReLU(),
		} 
		self.fc1 = nn.Linear(input_size, hidden_size_1)
		self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.fc3 = nn.Linear(hidden_size_2, output_size)
		# set activation
		self.activate = self.dic_activation[activate]

	def forward(self, x,device):
		out = self.fc1(x.to(device))
		out = self.activate(out)
		out = self.fc2(out)
		out = self.activate(out)
		out = self.fc3(out)
		return out 
