import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import os

import utils

from models.mnist_model import Generator, Discriminator, DHead, QHead
from dataloader import get_data
from utils import *
from config import params

import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(description='Train InfoGAN')
parser.add_argument('--exp_name', type=str, help='name of experiment')
parser.add_argument('--info_weight', type=float, help='weight of info_loss')
parser.add_argument('--tc_weight', type=float, help='weight of tc_loss')
args = parser.parse_args()

if os.path.exists(args.exp_name):
    # raise "experiment folder already exist!"
    pass
else:
    os.mkdir(args.exp_name)



if(params['dataset'] == 'MNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'CMNIST'):
    from models.cmnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'SVHN'):
    from models.svhn_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'CelebA'):
    from models.celeba_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif (params['dataset'] == 'dSprites'):
    # from models.dsprite_model import Generator, Discriminator, DHead, QHead
    from models.dsprite_model_modified import Generator, Discriminator, DHead, QHead, TC_Discriminator
elif (params['dataset'] == 'doubleMNIST'):
    from models.mmnist_model import Generator, Discriminator, DHead, QHead


# Set random seed for reproducibility.
seed = 1126
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

if params['dataset'] in ['MNIST', 'SVHN', 'FashionMNIST', 'CelebA']:
    dataloader = get_data(params['dataset'], params['batch_size'])
elif params['dataset'] == 'CMNIST':
    dataset = utils.ColoredMNIST()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
elif params['dataset'] == 'dSprites':
    dataset = utils.dSprites()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
elif params['dataset'] == 'doubleMNIST':
    dataset = utils.doubleMNIST()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
elif(params['dataset'] == 'SVHN'):
    params['num_z'] = 124
    params['num_dis_c'] = 4
    params['dis_c_dim'] = 10
    params['num_con_c'] = 4
elif(params['dataset'] == 'CelebA'):
    params['num_z'] = 128
    params['num_dis_c'] = 10
    params['dis_c_dim'] = 10
    params['num_con_c'] = 0
elif(params['dataset'] == 'FashionMNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
elif(params['dataset'] == 'dSprites'):
    params['num_z'] = 5
    params['num_dis_c'] = 0
    params['dis_c_dim'] = 3
    params['num_con_c'] = 5
elif(params['dataset'] == 'doubleMNIST'):
    params['num_z'] = 124
    params['num_dis_c'] = 2
    params['dis_c_dim'] = 10
    params['num_con_c'] = 4
elif(params['dataset'] == 'CMNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 3


gen_input_dim = params['num_z'] + params['num_dis_c'] * params['dis_c_dim'] + params['num_con_c']
netQ_dis_n = params['num_dis_c'] * params['dis_c_dim']
netQ_cont_n = params['num_con_c']

# in case of any 0, avoid error in model architecture
if netQ_dis_n == 0:
    netQ_dis_n = 1
if netQ_cont_n == 0:
    netQ_cont_n = 1


###########################
# Plot the training images.
###########################
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('{}/Training Images {}'.format(args.exp_name, params['dataset']))
plt.close('all')


#########################
# Initialise the network.
#########################
netG = Generator(input_dim=gen_input_dim).to(device)
netG.apply(weights_init)
print(netG)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

netD = DHead().to(device)
netD.apply(weights_init)
print(netD)

netQ = QHead(dis_n=netQ_dis_n, cont_n=netQ_cont_n).to(device)
netQ.apply(weights_init)
print(netQ)

tc_d = TC_Discriminator(z_dim=params['num_con_c']).to(device)
tc_d.apply(weights_init)
print(tc_d)


criterionD = nn.BCELoss()
criterionQ_dis = nn.CrossEntropyLoss()
criterionQ_con = NormalNLLLoss()

optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['lr_d'], betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['lr_g'], betas=(params['beta1'], params['beta2']))
optimTCD = optim.Adam([{'params': tc_d.parameters()}], lr=params['lr_tcd'], betas=(params['beta1'], params['beta2']))

#############################################
# Fixed Noise - for checkpoint sampling only
#############################################
z = torch.randn(params['dis_c_dim']*10, params['num_z'], 1, 1, device=device)
fixed_noise = z

if(params['num_dis_c'] != 0):
    idx = np.arange(params['dis_c_dim']).repeat(10)
    dis_c = torch.zeros(params['dis_c_dim']*10, params['num_dis_c'], params['dis_c_dim'], device=device)
    for i in range(params['num_dis_c']):
        dis_c[torch.arange(0, params['dis_c_dim']*10), i, idx] = 1.0

    dis_c = dis_c.view(params['dis_c_dim']*10, -1, 1, 1)
    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if(params['num_con_c'] != 0):
    con_c = torch.rand(params['dis_c_dim']*10, params['num_con_c'], 1, 1, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)



########################
# Prep before training
########################

real_label = 1
fake_label = 0

# List variables to store results of training.
img_list = []
G_losses = []
D_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print('params', params)
print("-"*25)

start_time = time.time()
iters = 0


########################
# Start training
########################
for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(dataloader, 0):
        # print('data shape', data.shape)

        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)

        ##################################
        # Train discriminator and DHead
        ##################################
        optimD.zero_grad()
        # Real data
        label = torch.full((b_size, ), real_label, device=device).float()
        if params['label_smoothing']:
            label = ((1.2 - 0.7) * torch.rand((b_size, )) + 0.7).to(device)
        output1 = discriminator(real_data)

        # print('output1', output1.shape)
        probs_real = netD(output1).view(-1).float()
        # print('probs_real, label', probs_real.shape, label.shape)
        loss_real = criterionD(probs_real, label)
        # Calculate gradients.
        loss_real.backward()

        # Fake data
        label.fill_(fake_label)
        noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device)
        fake_data = netG(noise)
        output2 = discriminator(fake_data.detach())

        probs_fake = netD(output2).view(-1)
        # print('probs_fake, label', probs_fake.shape, label.shape)
        loss_fake = criterionD(probs_fake, label)
        # Calculate gradients.
        loss_fake.backward()

        # Net Loss for the discriminator
        D_loss = loss_real + loss_fake
        # Update parameters
        optimD.step()

        ##################################
        # Train Generator and QHead
        ##################################

        optimG.zero_grad()

        # Fake data treated as real.
        output = discriminator(fake_data)
        label.fill_(real_label)
        if params['label_smoothing']:
            label = ((1.2 - 0.7) * torch.rand((b_size,)) + 0.7).to(device)
        probs_fake = netD(output).view(-1)
        gen_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = netQ(output)
        target = torch.LongTensor(idx).to(device)
        # Calculating loss for discrete latent code.
        dis_loss = 0
        for j in range(params['num_dis_c']):
            dis_loss += criterionQ_dis(q_logits[:, j*params['dis_c_dim'] : (j + 1) * params['dis_c_dim']], target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if (params['num_con_c'] != 0):
            con_loss = criterionQ_con(noise[:, params['num_z'] + params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1




        ################
        # Add TC Loss
        ################
        logvar = torch.log(q_var)
        z_recon = reparametrize(q_mu, logvar)
        # print('z_recon', z_recon.shape)
        D_z = tc_d(z_recon)
        # print('D_z', D_z.shape)
        # tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
        tc_loss = (D_z[:, 0] - D_z[:, 1]).mean()

        # Net loss for generator.
        G_loss = gen_loss + dis_loss + args.info_weight * con_loss + args.tc_weight * tc_loss

        G_loss.backward(retain_graph=True)

        #########################
        # Train TC Discriminator
        #########################
        # print('Train TC Discriminator')
        optimTCD.zero_grad()
        noise2, _ = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'],
                                     b_size, device)

        fake_data2 = netG(noise2)
        output_tc = discriminator(fake_data2)
        _, q_mu2, q_var2 = netQ(output_tc)

        # print('q_logits, q_mu, q_var', q_logits.shape, q_mu.shape, q_var.shape)

        z_prime = reparametrize(q_mu2, q_var2)
        z_pperm = permute_dims(z_prime).detach()

        D_z_pperm = tc_d(z_pperm)
        # print('D_z_pperm', D_z_pperm.shape, D_z_pperm)

        ones = torch.ones(b_size, dtype=torch.long, device=device)
        zeros = torch.zeros(b_size, dtype=torch.long, device=device)

        D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

        print('gen_loss: {}, con_loss: {}, tc_loss: {}, G_loss: {}, D_tc_loss: {}'.format(gen_loss, con_loss, tc_loss, G_loss, D_tc_loss))

        D_tc_loss.backward()

        optimG.step()
        optimTCD.step()



        #############
        # Print Log
        #############
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_TC_D: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader),
                    D_loss.item(), G_loss.item(), D_tc_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        iters += 1

    #################
    # End of an epoch
    #################
    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Save checkpoint samples
    if((epoch+1) == 1 or (epoch+1) % 1 == 0):
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("{}/Epoch_%d {}".format(args.exp_name, params['dataset']) %(epoch+1))
        plt.close('all')

    # Save checkpoint models
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'netG' : netG.state_dict(),
            'discriminator' : discriminator.state_dict(),
            'netD' : netD.state_dict(),
            'netQ' : netQ.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
            'params' : params
            }, '{}/model_epoch_%d_{}'.format(args.exp_name, params['dataset']) %(epoch+1))

####################
# Training finished
####################
training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)

# Generate image to check performance of trained generator.
with torch.no_grad():
    gen_data = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig("{}/Epoch_%d_{}".format(args.exp_name, params['dataset']) %(params['num_epochs']))

# Save network weights.
torch.save({
    'netG' : netG.state_dict(),
    'discriminator' : discriminator.state_dict(),
    'netD' : netD.state_dict(),
    'netQ' : netQ.state_dict(),
    'optimD' : optimD.state_dict(),
    'optimG' : optimG.state_dict(),
    'params' : params
    }, '{}/model_final_{}'.format(args.exp_name, params['dataset']))


# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("{}/Loss Curve {}".format(args.exp_name, params['dataset']))

# Animation showing the improvements of the generator.
# fig = plt.figure(figsize=(10,10))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# anim.save('infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
# plt.show()