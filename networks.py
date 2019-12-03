import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
class quadexp(nn.Module):
	def __init__(self, sigma = 2.):
		super(quadexp,self).__init__()
		self.sigma = sigma
	def forward(self,x):
		return tr.exp(-x**2/(self.sigma**2))

class NoisyLinear(nn.Linear):
	def __init__(self, in_features, out_features, noise_level=1., noise_decay = 0.1, bias=False):
		super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
		self.noise_level = noise_level
		self.register_buffer("epsilon_weight", tr.zeros(out_features, in_features))
		if bias:
			self.register_buffer("epsilon_bias", tr.zeros(out_features))
		self.noisy_mode = False
		self.noise_decay = noise_decay

	def update_noise_level(self):
		self.noise_level = self.noise_decay * self.noise_level
	def set_noisy_mode(self,is_noisy):
		self.noisy_mode = is_noisy

	def forward(self, input):
		if self.noisy_mode:
			tr.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
			bias = self.bias
			if bias is not None:
				tr.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
				bias = bias + self.noise_level * Variable(self.epsilon_bias, requires_grad  = False)
			self.noisy_mode = False
			return F.linear(input, self.weight + self.noise_level * Variable(self.epsilon_weight, requires_grad=False), bias)
		else:
			return F.linear(input, self.weight , self.bias)
	def add_noise(self):
		tr.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
		self.weight.data +=  self.noise_level * Variable(self.epsilon_weight, requires_grad=False)
		bias = self.bias
		if bias is not None:
			tr.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
			self.bias.data += self.noise_level * Variable(self.epsilon_bias, requires_grad  = False)

class OneHiddenLayer(nn.Module):
	def __init__(self,d_int, H, d_out,non_linearity = quadexp(),bias=False):
		super(OneHiddenLayer,self).__init__()
		self.linear1 = tr.nn.Linear(d_int, H,bias=bias)
		self.linear2 = tr.nn.Linear(H, d_out,bias=bias)
		self.non_linearity = non_linearity
		self.d_int = d_int
		self.d_out = d_out

	def weights_init(self,center, std):
		self.linear1.weights_init(center,std)
		self.linear2.weights_init(center,std)


	def forward(self, x):
		h1_relu = self.linear1(x).clamp(min=0)
		h2_relu = self.linear2(h1_relu)
		h2_relu = self.non_linearity(h2_relu)

		return h2_relu


class NoisyOneHiddenLayer(nn.Module):
	def __init__(self,d_int, H, d_out, n_particles,non_linearity = quadexp(),noise_level=1., noise_decay = 0.1,bias=False):
		super(NoisyOneHiddenLayer,self).__init__()

		self.linear1 = NoisyLinear(d_int, H*n_particles,noise_level = noise_level,noise_decay=noise_decay,bias=bias)
		self.linear2 = NoisyLinear(H*n_particles, n_particles*d_out,noise_level = noise_level,noise_decay=noise_decay,bias= bias)

		self.non_linearity = non_linearity
		self.n_particles = n_particles
		self.d_out = d_out

	def set_noisy_mode(self,is_noisy):
		self.linear1.set_noisy_mode(is_noisy)
		self.linear2.set_noisy_mode(is_noisy)

	def update_noise_level(self):
		self.linear1.update_noise_level()
		self.linear2.update_noise_level()

	def weights_init(self,center, std):
		self.linear1.weights_init(center,std)
		self.linear2.weights_init(center,std)

	def forward(self, x):
		h1_relu = self.linear1(x).clamp(min=0)
		h2_relu = self.linear2(h1_relu)
		h2_relu = h2_relu.view(-1,self.d_out, self.n_particles)
		h2_relu = self.non_linearity(h2_relu)

		return h2_relu
	def add_noise(self):
		self.linear1.add_noise()
		self.linear2.add_noise()

class SphericalTeacher(tr.utils.data.Dataset):

	def __init__(self,network, N_samples, dtype, device):
		D = network.d_int
		self.device = device
		self.source = tr.distributions.multivariate_normal.MultivariateNormal(tr.zeros(D ,dtype=dtype,device=device), tr.eye(D,dtype=dtype,device=device))
		source_samples = self.source.sample([N_samples])
		inv_norm = 1./tr.norm(source_samples,dim=1)
		self.X = tr.einsum('nd,n->nd',source_samples,inv_norm)
		self.total_size = N_samples
		self.network = network

		with tr.no_grad():
			self.Y = self.network(self.X)

	def __len__(self):
		return self.total_size 
	def __getitem__(self,index):
		return self.X[index,:],self.Y[index,:]

