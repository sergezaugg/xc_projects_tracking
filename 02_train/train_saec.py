#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import torch
from train_saec.tools import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mod_dir = "D:/xc_real_projects/pytorch_models"
mod_dir = "D:/xc_real_projects/pytorch_models/conv_tran_texture_01"

dat_tra_dir = "D:/xc_real_projects/xc_data_01_train/images_24000sps_20250707_231415"
dat_tes_dir = "D:/xc_real_projects/xc_data_02_Corvidae/images_24000sps_20250708_091750"





# Create the cold (=random init) instances of the models
mca = MakeColdAutoencoders(dir_models = mod_dir)
mod_arch = mca.make()
mod_arch.keys()
# 'conv_tran_L5_TP32_ch256', 'conv_tran_L5_sym', 'conv_tran_L5_TP32_ch512', 'conv_conv_L5_TP32', 'conv_tran_L4_TP08', 'conv_tran_texture_01', 'conv_tran_textr_resh'] 
mod_arch['conv_tran_texture_01']
mod_arch['conv_tran_textr_resh']

# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(
	data_gen = 'daugm_denoise', 
	dir_models = mod_dir, 
	dir_train_data = dat_tra_dir, 
    dir_test_data = dat_tes_dir,
	hot_start = False, 
    # model_tag = "conv_tran_L5_TP32", 
	# model_tag = "conv_tran_L4_TP08",
	# model_tag = "conv_tran_L5_sym",
	model_tag = "conv_tran_texture_01",
	device = device
	)

# Directly check data augmentation
at.make_data_augment_examples().show()

# Start training (.pth files will be saved to disk)
_, _, tstmp01 = at.train_autoencoder(n_epochs = 1, batch_size_tr = 8, batch_size_te = 32, devel = False)





dat_tes_dir_b = "D:/xc_real_projects/xc_hand_selected"
er = EvaluateReconstruction(dir_models = mod_dir, device = device)
er.evaluate_reconstruction_on_examples(path_images = dat_tes_dir_b, time_stamp_model = tstmp01, n_images = 23, shuffle = False).show()




# tstmp01 = '20250712_235352'

# resume
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
_, _, tstmp01 = at.train_autoencoder(n_epochs = 2, batch_size_tr = 8, batch_size_te = 32, devel = False)




#-----------------------------------------------------------
# assess results 
import os 
import torch
from train_saec.tools import EvaluateReconstruction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dat_tes_dir_b = "D:/xc_real_projects/xc_hand_selected"

# mod_dir_b = "D:/xc_real_projects/pytorch_models/conv_conv_L5_TP32"
# mod_dir_b = "D:/xc_real_projects/pytorch_models/conv_tran_L4_TP08"
# mod_dir_b = "D:/xc_real_projects/pytorch_models/conv_tran_L5_TP32_ch512"

mod_dir_b = "D:/xc_real_projects/pytorch_models/conv_tran_L5_TP32_ch256"
mod_dir_b = "D:/xc_real_projects/pytorch_models/conv_tran_L5_sym"
mod_dir_b = "D:/xc_real_projects/pytorch_models/conv_tran_texture_01"
mod_dir_b = "D:/xc_real_projects/pytorch_models/conv_tran_textr_resh"

mod_li = list(set([a[0:15] for a in os.listdir(mod_dir_b)]))
mod_li.sort()
# mod_li = mod_li[-1:]

for tstmp01 in mod_li:
	er = EvaluateReconstruction(dir_models = mod_dir_b, device = device)
	er.evaluate_reconstruction_on_examples(path_images = dat_tes_dir_b, time_stamp_model = tstmp01, n_images = 23, shuffle = False).show()








