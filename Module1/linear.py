import torch

'''
Complete this class by instantiating parameters called "self.weight" and "self.bias", and
use them to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomLinear(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super().__init__()
		# Initialize weight parameter
		inital_weights = 0.1*torch.randn(output_size, input_size)
		self.weight = torch.nn.Parameter(inital_weights)
		# Initialize bias parameter
		inital_bias = 0.1*torch.randn(output_size)
		self.bias = torch.nn.Parameter(inital_bias)

	def forward(self, x):
		'''
		x is a tensor contain a batch of vectors, size (B, input_size).
		This should return a tensor of size (B, output_size).
		'''
		# Perform linear transformation y = x @ W.T + bias
		return x @ self.weight.T + self.bias