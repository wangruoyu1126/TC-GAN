#######
# MWS #
#######

# dsprites
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.1_tcw_0.001 --info_weight 0.1 --tc_weight 0.001
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.1_tcw_0.005 --info_weight 0.1 --tc_weight 0.005
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.1_tcw_0.01 --info_weight 0.1 --tc_weight 0.01
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.1_tcw_0.02 --info_weight 0.1 --tc_weight 0.02
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.1_tcw_0.03 --info_weight 0.1 --tc_weight 0.03

CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.2_tcw_0.001 --info_weight 0.2 --tc_weight 0.001
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.2_tcw_0.005 --info_weight 0.2 --tc_weight 0.005
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.2_tcw_0.01 --info_weight 0.2 --tc_weight 0.01
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.2_tcw_0.02 --info_weight 0.2 --tc_weight 0.02
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dsprites_0_5_5_mws_infow_0.2_tcw_0.03 --info_weight 0.2 --tc_weight 0.03


#######
# DRT #
#######
conda activate torch && cd code/InfoTCGAN && clear
# dsprites
CUDA_VISIBLE_DEVICES=0 python train_factorvae.py --exp_name dsprites_0_5_5_drt_infow_0.1_tcw_0.00001 --info_weight 0.1 --tc_weight 0.00001
CUDA_VISIBLE_DEVICES=0 python train_factorvae.py --exp_name dsprites_0_5_5_drt_infow_0.1_tcw_0.00002 --info_weight 0.1 --tc_weight 0.00002
CUDA_VISIBLE_DEVICES=0 python train_factorvae.py --exp_name dsprites_0_5_5_drt_infow_0.1_tcw_0.00005 --info_weight 0.1 --tc_weight 0.00005
CUDA_VISIBLE_DEVICES=0 python train_factorvae.py --exp_name dsprites_0_5_5_drt_infow_0.1_tcw_0.00008 --info_weight 0.1 --tc_weight 0.00008
CUDA_VISIBLE_DEVICES=0 python train_factorvae.py --exp_name dsprites_0_5_5_drt_infow_0.1_tcw_0.0001 --info_weight 0.1 --tc_weight 0.0001





##############
# doubleMNIST 
##############
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dmnist_20_4_124_mws_infow_0.1_tcw_0 --info_weight 0.1 --tc_weight 0
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dmnist_20_4_124_mws_infow_0.1_tcw_0.001 --info_weight 0.1 --tc_weight 0.001
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dmnist_20_4_124_mws_infow_0.1_tcw_0.005 --info_weight 0.1 --tc_weight 0.005
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dmnist_20_4_124_mws_infow_0.1_tcw_0.01 --info_weight 0.1 --tc_weight 0.01
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dmnist_20_4_124_mws_infow_0.1_tcw_0.02 --info_weight 0.1 --tc_weight 0.02
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name dmnist_20_4_124_mws_infow_0.1_tcw_0.03 --info_weight 0.1 --tc_weight 0.03




##############
# 3dshapes 
##############
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name 3dshapes_0_5_5_mws_infow_0.1_tcw_0.001 --info_weight 0.1 --tc_weight 0.001
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name 3dshapes_0_5_5_mws_infow_0.1_tcw_0.005 --info_weight 0.1 --tc_weight 0.005
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name 3dshapes_0_5_5_mws_infow_0.1_tcw_0.01 --info_weight 0.1 --tc_weight 0.01
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name 3dshapes_0_5_5_mws_infow_0.1_tcw_0.02 --info_weight 0.1 --tc_weight 0.02
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name 3dshapes_0_5_5_mws_infow_0.1_tcw_0.03 --info_weight 0.1 --tc_weight 0.03




###############
# FashionMNIST
###############
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name fmnist_10_2_62_mws_infow_0.1_tcw_0_test1 --info_weight 1 --tc_weight 0
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name fmnist_10_2_62_mws_infow_0.1_tcw_0_test2 --info_weight 1 --tc_weight 0


CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name fmnist_10_2_62_mws_infow_0.1_tcw_0.001 --info_weight 0.1 --tc_weight 0.001
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name fmnist_10_2_62_mws_infow_0.1_tcw_0.005 --info_weight 0.1 --tc_weight 0.005
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name fmnist_10_2_62_mws_infow_0.1_tcw_0.01 --info_weight 0.1 --tc_weight 0.01
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name fmnist_10_2_62_mws_infow_0.1_tcw_0.02 --info_weight 0.1 --tc_weight 0.02
CUDA_VISIBLE_DEVICES=0 python train_btcvae.py --exp_name fmnist_10_2_62_mws_infow_0.1_tcw_0.03 --info_weight 0.1 --tc_weight 0.03





