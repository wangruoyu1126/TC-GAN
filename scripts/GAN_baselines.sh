##############
# InfoGAN_CR #
##############
conda activate torch && cd code/InfoGAN_CR_torch
CUDA_VISIBLE_DEVICES=0 python src/main.py --use_visdom False





##########
# IB_GAN #
##########
CUDA_VISIBLE_DEVICES= python main.py configs/config_dsprite.json
CUDA_VISIBLE_DEVICES= python main.py configs/config_3dchairs.json
CUDA_VISIBLE_DEVICES= python main.py configs/config_celeba.json






