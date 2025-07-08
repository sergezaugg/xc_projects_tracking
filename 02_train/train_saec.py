#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import torch
from train_saec.tools import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mod_dir = "D:/xc_real_projects/pytorch_models"
dat_tra_dir = "D:/xc_real_projects/xc_data_01_train/images_24000sps_20250707_231415"
dat_tes_dir = "D:/xc_real_projects/xc_data_02_Corvidae/images_24000sps_20250708_091750"

# Create the cold (=random init) instances of the models
mca = MakeColdAutoencoders(dir_models = mod_dir)
mod_arch = mca.make()
mod_arch.keys()
mod_arch['GenC_new_TP32_CH0256']
# mod_arch['GenB3blocks']
mod_arch['GenBTP32_CH0256']



# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(
	data_gen = 'daugm_denoise', 
	dir_models = mod_dir, 
	dir_train_data = dat_tra_dir, 
    dir_test_data = dat_tes_dir,
	hot_start = False, 
    model_tag = "GenC_new_TP32_CH0256", 
	device = device
	)

# Directly check data augmentation
at.make_data_augment_examples().show()

# Start training (.pth files will be saved to disk)
_, _, tstmp01 = at.train_autoencoder(n_epochs = 5, batch_size_tr = 8, batch_size_te = 32, devel = False)

# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(
	data_gen = 'daugm_denoise', 
    dir_models = mod_dir,
	dir_train_data = dat_tra_dir, 
    dir_test_data = dat_tes_dir,
    hot_start = True, 
	model_tag = tstmp01, 
	device = device
	)

# Directly check data augmentation
at.make_data_augment_examples().show()

# Start training (.pth files will be saved to disk)
_, _, tstmp01 = at.train_autoencoder(n_epochs = 1, batch_size_tr = 32, batch_size_te = 32, devel = False)

tstmp01 = '20250708_132910'

# EvaluateReconstruction
er = EvaluateReconstruction(dir_models = mod_dir, device = device)
er.evaluate_reconstruction_on_examples(path_images = dat_tes_dir, time_stamp_model = tstmp01, n_images = 32, shuffle = False).show()







