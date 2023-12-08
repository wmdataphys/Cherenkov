import os
import json
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataloader.dataloader import CreateLoaders
from dataloader.dataset import create_dataset
from pickle import dump
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import pandas as pd
from pickle import load
from models.model_v2 import cVAE_DIRC
import torch.nn.functional as F
from torch.utils.data import Subset
from utils.utils import mmd_loss_function,FocalLoss

def main(config,resume):

	# Setup random seed
	torch.manual_seed(config['seed'])
	np.random.seed(config['seed'])
	random.seed(config['seed'])
	torch.cuda.manual_seed(config['seed'])

	# Create experiment name
	curr_date = datetime.now()
	exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
	exp_name = exp_name[:-11]
	print(exp_name)

	# Create directory structure
	output_folder = config['output']['dir']
	os.mkdir(os.path.join(output_folder,exp_name))
	with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
		json.dump(config, outfile)


	# Load the dataset
	print('Creating Loaders.')
	#full_dataset = DIRC_Dataset(root_dir=config['dataset']['root_dir'],data_file=config['dataset']['data_filename'],cond_file=config['dataset']['cond_filename'],label_file=config['dataset']['label_filename'])
	full_dataset = create_dataset(config)
	history = {'train_loss':[],'val_loss':[],'lr':[]}
	run_val = True
	train_loader,val_loader,test_loader = CreateLoaders(full_dataset,config)
	print("Training Size: {0}".format(len(train_loader.dataset)))
	print("Validation Size: {0}".format(len(val_loader.dataset)))
	print("Testing Size: {0}".format(len(test_loader.dataset)))



	 # Create the model
	#net = cVAE_DIRC((2, 48, 144), 2, nhid = 16, ncond = 5)
	net = cVAE_DIRC(nhid=16,ncond=5,classification=config['classification'])
	t_params = sum(p.numel() for p in net.parameters())
	print("Network Parameters: ",t_params)
	net.to('cuda')

	print(net)


	# Optimizer
	lr = float(config['optimizer']['lr'])
	step_size = int(config['lr_scheduler']['step_size'])
	gamma = float(config['lr_scheduler']['gamma'])
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	num_epochs=int(config['num_epochs'])

	startEpoch = 0
	global_step = 0

	if resume:
		print('===========  Resume training  ==================:')
		#dict = torch.load(resume)
		dict = np.load(resume,allow_pickle=True)
		net.load_state_dict(dict['net_state_dict'])
		optimizer.load_state_dict(dict['optimizer_state_dict'])
		scheduler.load_state_dict(dict['scheduler'])
		startEpoch = dict['epoch']+1
		#history = dict['history']
		#global_step = dict['global_step']

		print('       ... Start at epoch:',startEpoch)
	else:
		print("========= Starting Training ================:")

	print('===========  Optimizer  ==================:')
	print('      LR:', lr)
	print('      step_size:', step_size)
	print('      gamma:', gamma)
	print('      num_epochs:', num_epochs)
	print('')

	# Train


	# Define your loss function
	#classification_loss = torch.nn.CrossEntropyLoss()
	classification_loss = nn.BCELoss()
	pixel_wise_rec = FocalLoss(gamma=2,size_average=False)#nn.BCEWithLogitsLoss(reduction='mean')
	#pixel_wise_time = torch.nn.MSELoss()
	pixel_wise_time = nn.SmoothL1Loss(reduction='mean')
	#loss_fn = torch.nn.NLLLoss(reduction='sum')
	#loss_fn = torch.nn.MSELoss()
	alpha = 1

	for epoch in range(startEpoch,num_epochs):

		kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

		###################
		## Training loop ##
		###################
		net.train()
		running_loss = 0.0

		for i, data in enumerate(train_loader):
			x  = data[0].to('cuda').float()
			y  = data[1].to('cuda').float()
			label = data[2].to('cuda').float().unsqueeze(0)
			#print(label.shape)
			# reset the gradient
			optimizer.zero_grad()

			# forward pass, enable to track our gradient
			with torch.set_grad_enabled(True):
				if bool(config['classification']):
					recon,z,pred = net(x,y)
					pred = pred.transpose(0,1)
				else:
					recon,z = net(x,y)
			#print(pred.shape)

			mmd = mmd_loss_function(recon, x, z, gaussian_sigma=config['optimizer']['var_MMD'])
			# pixel_wise_rec only utilizes the "hits channel", include MSE for time axis.
			reconstruction = (pixel_wise_rec(recon,x) + pixel_wise_time(recon[:,1,:,:],x[:,1,:,:]))#*config['optimizer']['NLL_weight']
			mmd = mmd*100.#config['optimizer']['MMD_weight']
			loss = reconstruction + mmd
			if bool(config['classification']):
				ll = classification_loss(pred, label)*config['optimizer']['CE_weight']
				loss += ll
				train_acc = (torch.sum(torch.round(pred) == label)).item() / label.shape[1]
			# backprop
			loss.backward()
			optimizer.step()

			# statistics
			running_loss += loss.item() * label.shape[0]

			#print(label.shape)
			if bool(config['classification']):
				kbar.update(i, values=[("loss", loss.item()),("Rec_loss",reconstruction.item()),("MMD_loss",mmd.item()),("BCE",ll.item()),("Train_Acc",train_acc)])
			else:
				kbar.update(i, values=[("loss", loss.item()),("Rec_loss",reconstruction.item()),("MMD_loss",mmd.item())])

			global_step += 1


		scheduler.step()

		history['train_loss'].append(running_loss / len(train_loader.dataset))
		history['lr'].append(scheduler.get_last_lr()[0])


		######################
		## validation phase ##
		######################
		if run_val:
			net.eval()
			val_loss = 0.0
			val_rec_loss = 0.0
			val_mmd = 0.0
			val_ll = 0.0
			val_acc = 0.0
			for i, data in enumerate(val_loader):
				x  = data[0].to('cuda').float()
				y  = data[1].to('cuda').float()
				label = data[2].to('cuda').float().unsqueeze(0)

				with torch.no_grad():
					if bool(config['classification']):
						recon,z,pred = net(x,y)
						pred = pred.transpose(0,1)
					else:
						recon,z = net(x,y)
				#print(pred.shape)

				mmd = mmd_loss_function(recon, x, z, gaussian_sigma=config['optimizer']['var_MMD'])
				reconstruction = pixel_wise_rec(recon,x) + pixel_wise_time(recon[:,1,:,:],x[:,1,:,:])
				val_rec_loss += reconstruction*config['optimizer']['NLL_weight']
				val_mmd += mmd*config['optimizer']['MMD_weight']
				#loss = reconstruction + mmd
				if bool(config['classification']):
					val_ll += classification_loss(pred, label)*config['optimizer']['CE_weight']
					val_acc += (torch.sum(torch.round(pred) == label)).item() / label.shape[1]
				#loss += ll
				# backprop

				# statistics

			#print(label.shape)
			if bool(config['classification']):
				val_acc = val_acc/len(val_loader)
				val_loss = (val_rec_loss + val_mmd + val_ll)/len(val_loader)

				kbar.add(1, values=[("Val_loss", val_loss.item()),("Val_Acc",val_acc)])

			else:
				val_loss = (val_rec_loss + val_mmd)/len(val_loader)
				kbar.add(1, values=[("Val_loss", val_loss.item()),("Val_mmd",val_mmd/len(val_loader)),("Val_Rec",val_rec_loss/len(val_loader))])


			#kbar.add(1, values=[("val_loss" ,val_loss),("val_reg_loss",val_rec_loss.item()),("val_kl_loss",val_kl_div.item())])

			# Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
			name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.6f}_val_acc_{:.6f}.pth'.format(epoch, val_loss,val_acc)

		else:
			kbar.add(1,values=[('val_loss',0.)])
			name_output_file = config['name']+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

		filename = os.path.join(output_folder , exp_name , name_output_file)

		checkpoint={}
		checkpoint['net_state_dict'] = net.state_dict()
		checkpoint['optimizer'] = optimizer.state_dict()
		checkpoint['scheduler'] = scheduler.state_dict()
		checkpoint['epoch'] = epoch
		checkpoint['history'] = history
		checkpoint['global_step'] = global_step

		torch.save(checkpoint,filename)

		print('')




if __name__=='__main__':
	# PARSE THE ARGS
	parser = argparse.ArgumentParser(description='Hackaton Training')
	parser.add_argument('-c', '--config', default='config.json',type=str,
						help='Path to the config file (default: config.json)')
	parser.add_argument('-r', '--resume', default=None, type=str,
						help='Path to the .pth model checkpoint to resume training')
	args = parser.parse_args()

	config = json.load(open(args.config))

	main(config,args.resume)
