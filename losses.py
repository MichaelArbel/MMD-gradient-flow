import torch as tr
import torch.nn as nn
from torch import autograd


class mmd2_noise_injection(autograd.Function):

	@staticmethod
	def forward(ctx,true_feature,fake_feature,noisy_feature):
		b_size,d, n_particles = noisy_feature.shape
		with  tr.enable_grad():

			mmd2 = tr.mean((true_feature-fake_feature)**2)
			mean_noisy_feature = tr.mean(noisy_feature,dim = -1 )

			mmd2_for_grad = (n_particles/b_size)*(tr.einsum('nd,nd->',fake_feature,mean_noisy_feature) - tr.einsum('nd,nd->',true_feature,mean_noisy_feature))

		ctx.save_for_backward(mmd2_for_grad,noisy_feature)

		return mmd2

	@staticmethod
	def backward(ctx, grad_output):
		mmd2_for_grad, noisy_feature = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2_for_grad, inputs=noisy_feature,
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0] 
				
		return None, None, gradients


class mmd2_func(autograd.Function):

	@staticmethod
	def forward(ctx,true_feature,fake_feature):

		b_size,d, n_particles = fake_feature.shape

		with  tr.enable_grad():

			mmd2 = (n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)

		ctx.save_for_backward(mmd2,fake_feature)

		return (1./n_particles)*mmd2

	@staticmethod
	def backward(ctx, grad_output):

		mmd2, fake_feature = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0] 
				
		return None, gradients


class sobolev(autograd.Function):
	@staticmethod
	def forward(ctx,true_feature,fake_feature,matrix):

		b_size,_, n_particles = fake_feature.shape

		m = tr.mean(fake_feature,dim=-1) -  true_feature

		alpha = tr.solve(m,matrix)[0].clone().detach()

		with  tr.enable_grad():

			mmd2 = (0.5*n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)
			mmd2_for_grad = (1./b_size)*tr.einsum('id,idm->',alpha,fake_feature)
		
		ctx.save_for_backward(mmd2_for_grad,fake_feature)

		return (1./n_particles)*mmd2

	@staticmethod
	def backward(ctx, grad_output):
		mmd2, fake_feature = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0] 
				
		return None, gradients,None


class MMD(nn.Module):
	def __init__(self,student,with_noise):
		super(MMD, self).__init__()
		self.student = student
		self.mmd2 = mmd2_noise_injection.apply
		self.with_noise=with_noise
	def forward(self,x,y):
		if self.with_noise:
			out = tr.mean(self.student(x),dim = -1).clone().detach()
			self.student.set_noisy_mode(True)
			noisy_out = self.student(x)
			loss = 0.5*self.mmd2(y,out,noisy_out)
		else:
			fake_feature = tr.mean(self.student(x),dim=-1)
			loss = 0.5*tr.mean((y-fake_feature)**2)
		return loss

class MMD_Diffusion(nn.Module):
	def __init__(self,student):
		super(MMD_Diffusion, self).__init__()
		self.student = student
		self.mmd2 = mmd2_func.apply
	def forward(self,x,y):
		self.student.add_noise()
		noisy_out = self.student(x)
		
		loss = 0.5*self.mmd2(y,noisy_out)
		return loss

class Sobolev(nn.Module):
	def __init__(self,student):
		super(Sobolev, self).__init__()
		self.student = student
		self.sobolev = sobolev.apply
		self.lmbda = 1e-6
	def forward(self,x,y):
		self.student.zero_grad()
		out = self.student(x)
		b_size,_,num_particles = out.shape
		grad_out = compute_grad(self.student,x)
		matrix = (1./(num_particles*b_size))*tr.einsum('im,jm->ij',grad_out,grad_out)+self.lmbda*tr.eye(b_size, dtype= x.dtype, device=x.device)
		matrix = matrix.clone().detach()
		loss = self.sobolev(y,out,matrix)
		return loss

def compute_grad(net,x):
	J = []
	F = net(x)
	F = tr.einsum('idm->i',F)
	b_size = F.shape[0]
	for i in range(b_size):
		if i==b_size-1:
			grads =  autograd.grad(F[i], net.parameters(),retain_graph=False)
		else:
			grads =  autograd.grad(F[i], net.parameters(),retain_graph=True)
		grads = [x.view(-1) for x in grads]
		grads = tr.cat(grads)
		J.append(grads)

	return tr.stack(J,dim=0)

