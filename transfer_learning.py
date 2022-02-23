from __future__ import print_function, division

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable 

import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os 

class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size_1)
		self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.fc3 = nn.Linear(hidden_size_2, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out 

class NeuralNet_sherpa_optimize(nn.Module):
	def __init__(self, input_size,output_size, Parameters):
		super(NeuralNet_sherpa_optimize, self).__init__()
		#unpack sherpa package note 2 properties used outside
		hidden_size_1=Parameters["H_l1"]
		hidden_size_2=Parameters["H_l2"]
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
		out = self.fc1(x)
		out = self.activate(out)
		out = self.fc2(out)
		out = self.activate(out)
		out = self.fc3(out)
		return out 