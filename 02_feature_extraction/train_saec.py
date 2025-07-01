



# pip install --upgrade C:\Users\sezau\Desktop\src\train_saec\dist\train_saec-0.0.5-py3-none-any.whl
# pip uninstall train_saec

import torch
from train_saec import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction
# from src.train_saec import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cold_dir = "C:/Users/sezau/Downloads/aaa"
hot_dir = "C:/Users/sezau/Downloads/aaa"

dat_tra_dir = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
dat_tes_dir = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"

# (run once) Create the cold (=random init) instances of the models
mca = MakeColdAutoencoders(dir_cold_models = cold_dir)
mod_arch = mca.make()

# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(data_gen = 'daugm_denoise', dir_cold_models = cold_dir, dir_hot_models = hot_dir,
						dir_train_data = dat_tra_dir, dir_test_data = dat_tes_dir,
						hot_start = False, model_tag = "GenBTP16_CH0256", device = device
						)

# Directly check data augmentation
at.make_data_augment_examples().show()

# Start training (.pth files will be saved to disk)
_, _, tstmp01 = at.train_autoencoder(n_epochs = 3, batch_size_tr = 8, batch_size_te = 32, devel = True)

# Or, initialize a AEC-trainer with a pre-trained model
at = AutoencoderTrain(data_gen = 'daugm_denoise', dir_cold_models = cold_dir, dir_hot_models = hot_dir,
						dir_train_data = dat_tra_dir, dir_test_data = dat_tes_dir,
						hot_start = True, model_tag = tstmp01, device = device
                        )

_, _, tstmp02 = at.train_autoencoder(n_epochs = 3, batch_size_tr = 8, batch_size_te = 32, devel = True)

# EvaluateReconstruction
er = EvaluateReconstruction(dir_hot_models = hot_dir, device = device)
er.evaluate_reconstruction_on_examples(path_images = dat_tes_dir, time_stamp_model = tstmp02, n_images = 32, shuffle = False).show()







