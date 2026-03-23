###########
# beta-VAE
###########
CUDA_VISIBLE_DEVICES=6 python main.py -x betaH_dsprites betaH_dsprites && 
CUDA_VISIBLE_DEVICES= python main.py -x betaH_chairs betaH_chairs && 
CUDA_VISIBLE_DEVICES= python main.py -x betaH_mnist betaH_mnist && 
CUDA_VISIBLE_DEVICES= python main.py -x betaH_doublemnist betaH_doublemnist && 
CUDA_VISIBLE_DEVICES= python main.py -x betaH_celeba betaH_celeba


###############
# Annealed-VAE
###############
CUDA_VISIBLE_DEVICES=6 python main.py -x betaB_dsprites betaB_dsprites && 
CUDA_VISIBLE_DEVICES= python main.py -x betaB_chairs betaB_chairs && 
CUDA_VISIBLE_DEVICES= python main.py -x betaB_mnist betaB_mnist && 
CUDA_VISIBLE_DEVICES= python main.py -x betaB_doublemnist betaB_doublemnist && 
CUDA_VISIBLE_DEVICES= python main.py -x betaB_celeba betaB_celeba


#############
# factor-VAE
#############
CUDA_VISIBLE_DEVICES=6 python main.py -x factor_dsprites factor_dsprites && 
CUDA_VISIBLE_DEVICES= python main.py -x factor_chairs factor_chairs && 
CUDA_VISIBLE_DEVICES= python main.py -x factor_mnist factor_mnist && 
CUDA_VISIBLE_DEVICES= python main.py -x factor_doublemnist factor_doublemnist && 
CUDA_VISIBLE_DEVICES= python main.py -x factor_celeba factor_celeba


#############
# beta-TCVAE
#############
CUDA_VISIBLE_DEVICES=6 python main.py -x btcvae_dsprites btcvae_dsprites && 
CUDA_VISIBLE_DEVICES= python main.py -x btcvae_chairs btcvae_chairs && 
CUDA_VISIBLE_DEVICES= python main.py -x btcvae_mnist btcvae_mnist && 
CUDA_VISIBLE_DEVICES= python main.py -x btcvae_doublemnist btcvae_doublemnist && 
CUDA_VISIBLE_DEVICES= python main.py -x btcvae_celeba btcvae_celeba




######
# Viz
######
CUDA_VISIBLE_DEVICES=0 python main_viz.py factor_dsprites all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py factor_chairs all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py factor_mnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py factor_doublemnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py factor_celeba all


CUDA_VISIBLE_DEVICES=0 python main_viz.py btcvae_dsprites all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py btcvae_chairs all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py btcvae_mnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py btcvae_doublemnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py btcvae_celeba all


CUDA_VISIBLE_DEVICES=0 python main_viz.py betaH_dsprites all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaH_chairs all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaH_mnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaH_doublemnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaH_celeba all


CUDA_VISIBLE_DEVICES=0 python main_viz.py betaB_dsprites all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaB_chairs all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaB_mnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaB_doublemnist all && 
CUDA_VISIBLE_DEVICES=0 python main_viz.py betaB_celeba all








