#--------------------             
# Author : Serge Zaugg
# Description : Batch feature extraction form IDNN models with constant frequency pooling
#--------------------

import os
import gc
import torch
from fe_idnn import IDNN_extractor
from fe_saec import SAEC_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path_source_images = "D:/xc_real_projects/xc_data_history/xc_sw_europe/xc_spectrograms"
# path_save_features = "D:/xc_real_projects/xc_data_history/xc_sw_europe/extracted_features"

# path_source_images = "D:/xc_real_projects/xc_data_history/xc_parus_01/xc_spectrograms"
# path_save_features = "D:/xc_real_projects/xc_data_history/xc_parus_01/extracted_features"

path_source_images = "D:/xc_real_projects/xc_data_02_Corvidae/xc_spectrograms"
path_save_features = "D:/xc_real_projects/xc_data_02_Corvidae/extracted_features"

if not os.path.exists(path_save_features):
    os.mkdir(path_save_features) 

basiz = 64


#-----------
# SAEC

# "D:\xc_real_projects\pytorch_models\conv_tran_L5_TP32_ch256"
# "D:\xc_real_projects\pytorch_models\conv_tran_L5_sym"
# "D:\xc_real_projects\pytorch_models\conv_tran_texture_01"
# "D:\xc_real_projects\pytorch_models\conv_tran_textr_resh"

# DEF conv_tran_L5_TP32_ch256 (success : done 20250715 - 21:06)
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_L5_TP32_ch256/20250710_223517_encoder_script_conv_tran_L5_TP32_epo006.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = None, verbose = True) 
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# NEW conv_tran_L5_sym (success : done 20250715 - 20:29)
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_L5_sym/20250711_154215_encoder_script_conv_tran_L5_sym_epo006.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = None, verbose = True) 
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# NEW conv_tran_textr_resh (success : done 20250715 - 19:26)
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_textr_resh/20250714_090709_encoder_script_conv_tran_textr_resh_epo006.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = None, verbose = True)
# to free max of memory the hard way: - close session and start new session + run code below 
# ae.load_full_features_from_npz(path_npz = os.path.join(path_save_features, "full_features_saec_20250714_090709.npz"))
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# NEW conv_tran_texture_01 (success : done 20250716 - 13:45)
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_texture_01/20250716_101902_encoder_script_conv_tran_texture_01_epo006.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = None, verbose = True) 
# to free max of memory the hard way: - close session and start new session + run code below 
# ae.load_full_features_from_npz(path_npz = os.path.join(path_save_features, "xxxxxx.npz"))
ae.X.shape
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  







#-----------
# ResNet50

n_batches = 500000000000

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer1.2.conv3")
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer2.3.conv3")
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer3.5.conv3")
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer4.2.conv3")
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

#-----------
# vgg16

fe = IDNN_extractor(model_tag = "vgg16")
fe.create("features.2") #  block 1
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "vgg16")
fe.create("features.7") #  block 2
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "vgg16")
fe.create("features.14") #  block 3
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "vgg16")
fe.create("features.21") #  block 4
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "vgg16")
fe.create("features.28") #  block 5
fe.extract(image_path = path_source_images, fe_save_path = path_save_features,  freq_pool = 4, batch_size = basiz, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

