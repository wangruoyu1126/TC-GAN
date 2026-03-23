import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from models.mnist_model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

exp_folder = '/'.join(args.load_path.split('/')[:-1])
epoch_num = args.load_path.split('/')[-1].split('_')[2]

n_dis = 1
dis_dim = 10
n_cont = 2
n_noise = 62


############
# Load model
############
state_dict = torch.load(args.load_path)
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
params = state_dict['params']

netG = Generator(input_dim=n_noise + n_dis * dis_dim + n_cont).to(device)
netG.load_state_dict(state_dict['netG'])
print(netG)


###########################
# Continuous Variables
###########################
c = np.linspace(-2, 2, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(100, 1, 1, 1, device=device)

# Continuous latent code.
c2 = torch.cat((c, zeros), dim=1)
c3 = torch.cat((zeros, c), dim=1)
print('c2, c3 shape', c2.shape, c3.shape)

####################
# Discrete Variables
####################
idx = np.arange(10).repeat(10)
dis_c = torch.zeros(100, 10, 1, 1, device=device)
dis_c[torch.arange(0, 100), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(100, -1, 1, 1)
print('c1 shape', c1.shape)

########
# Noise
########
z = torch.randn(100, n_noise, 1, 1, device=device)
print('z shape', z.shape)

###########################
# Concat all latent code
###########################
# To see variation along c2 (Horizontally) and c1 (Vertically)
noise1 = torch.cat((z, c1, c2), dim=1)
# To see variation along c3 (Horizontally) and c1 (Vertically)
noise2 = torch.cat((z, c1, c3), dim=1)
print('noise dimensions', noise1.shape, noise2.shape)

##################
# Generate image
##################
with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()
print(generated_img1.shape)
save_image(generated_img1, "{}/generated_img1_epoch_{}.png".format(exp_folder, epoch_num), nrow=10, normalize=True)
# # Display the generated image. s
# fig = plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.show()






with torch.no_grad():
    generated_img2 = netG(noise2).detach().cpu()
print(generated_img2.shape)
save_image(generated_img2, "{}/generated_img2_epoch_{}.png".format(exp_folder, epoch_num), nrow=10, normalize=True)
# Display the generated image.
# fig = plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.show()





# with torch.no_grad():
#     generated_img3 = netG(noise3).detach().cpu()
# print(generated_img3.shape)
# save_image(generated_img3, "{}/generated_img3.png".format(exp_folder), nrow=10, normalize=True)
# # # Display the generated image.
# # fig = plt.figure(figsize=(10, 10))
# # plt.axis("off")
# # plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
# # plt.show()
#









