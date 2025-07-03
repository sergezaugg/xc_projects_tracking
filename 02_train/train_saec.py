#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import torch
from train_saec.tools import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cold_dir = "D:/xc_real_projects/pytorch_cold_models"
hot_dir  = "D:/xc_real_projects/pytorch_hot_models"

dat_tra_dir = "D:/xc_real_projects/xc_all_4_pooled/images_short"
dat_tes_dir = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"

# Create the cold (=random init) instances of the models
mca = MakeColdAutoencoders(dir_cold_models = cold_dir)
mod_arch = mca.make()
mod_arch.keys()
mod_arch['GenC_new_TP32_CH0256']

# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(
	data_gen = 'daugm_denoise', 
	dir_cold_models = cold_dir, dir_hot_models = hot_dir,
	dir_train_data = dat_tra_dir, dir_test_data = dat_tes_dir,
	hot_start = False, model_tag = "GenC_new_TP32_CH0256", 
    # hot_start = True, model_tag = '20250701_143520', 
	device = device
	)

# Directly check data augmentation
at.make_data_augment_examples().show()

# Start training (.pth files will be saved to disk)
_, _, tstmp01 = at.train_autoencoder(n_epochs = 1, batch_size_tr = 32, batch_size_te = 32, devel = False)

# EvaluateReconstruction
er = EvaluateReconstruction(dir_hot_models = hot_dir, device = device)
er.evaluate_reconstruction_on_examples(path_images = dat_tes_dir, time_stamp_model = '20250701_165057', n_images = 32, shuffle = False).show()







