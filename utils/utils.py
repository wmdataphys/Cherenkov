import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_kernel(x, y):
	x_size = x.size(0)
	y_size = y.size(0)
	dim = x.size(1)
	x = x.unsqueeze(1).expand(x_size, y_size, dim)
	y = y.unsqueeze(0).expand(x_size, y_size, dim)
	d = (-(x - y).pow(2).mean(2)/float(dim)).exp().mean()
	#kernel_input = torch.exp(-(d.pow(2).mean(2)/float(dim)))
	return d


def compute_mmd(x, y):
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	mmd = x_kernel + y_kernel - 2*xy_kernel
	return mmd


def mmd_loss_function(recon_x, x, z, gaussian_sigma=1.0):
	true_samples = torch.randn(recon_x.size()[0], z.size()[-1], requires_grad = False).to(x.device) * gaussian_sigma
	mmd = compute_mmd(true_samples, z)
	#nll = torch.nn.functional.mse_loss(x, recon_x, reduction='mean')
	return mmd


class FocalLoss(nn.Module):
	"""
	Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
	on difficult foreground detections.
	"""

	def __init__(self, gamma=0, size_average=False):
	    super(FocalLoss, self).__init__()
	    self.gamma = gamma
	    self.size_average = size_average

	def forward(self, prediction, target):
		# get class probability
		#prediction = prediction[:, 0,:, :].contiguous().flatten()
		#target = target[:, 0,:, :].contiguous().flatten()
		prediction = prediction[:,0,:,:].contiguous().flatten()
		target = target[:,0,:,:].contiguous().flatten()
		pt = torch.where(target == 1.0, prediction, 1-prediction)

		# compute focal loss
		loss = -1 * (1-pt)**self.gamma * torch.log(pt+1e-6)

		if self.size_average:
		    return loss.mean()
		else:
		    return loss.sum()
