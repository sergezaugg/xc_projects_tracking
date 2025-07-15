#--------------------             
# Author : Serge Zaugg
# Description : Batch feature extraction form IDNN models with constant frequency pooling
#--------------------

import os
import gc
import torch
# from fe_idnn import IDNN_extractor
from fe_saec import SAEC_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path_source_images = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
# path_save_features = "C:/Users/sezau/Downloads/xc_sw_europe"

# path_source_images = "D:/xc_real_projects/xc_data_02_Corvidae/images_24000sps_20250708_091750"
# path_save_features = "C:/Users/sezau/Downloads/xc_corvus_corax"


# path_source_images = "D:/xc_real_projects/xc_data_history/xc_parus_01/xc_spectrograms"
# path_save_features = "D:/xc_real_projects/xc_data_history/xc_parus_01/extracted_features"

path_source_images = "D:/xc_real_projects/xc_data_02_Corvidae/images_24000sps_20250708_091750"
path_save_features = "D:/xc_real_projects/xc_data_02_Corvidae/extracted_features"



if not os.path.exists(path_save_features):
    os.mkdir(path_save_features) 

# n_batches = 5
n_batches = 1000000000

basiz = 64



#-----------
# ResNet50

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

#-----------
# SAEC

# DEF good models 
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_L5_TP32_ch256/20250710_223517_encoder_script_conv_tran_L5_TP32_epo006.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = n_batches) 
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# NEW conv_tran_L5_sym
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_L5_sym/20250711_154215_encoder_script_conv_tran_L5_sym_epo006.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = n_batches) 
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# NEW conv_tran_texture_01
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_texture_01/20250713_224421_encoder_script_conv_tran_texture_01_epo004.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = n_batches) 
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  









# NEW conv_tran_textr_resh
path_model = "D:/xc_real_projects/pytorch_models/conv_tran_textr_resh/20250713_232131_encoder_script_conv_tran_textr_resh_epo003.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
ae.extract(image_path = path_source_images, fe_save_path = path_save_features, batch_size = basiz, shuffle = True , n_batches = n_batches, verbose = True)

# to free max of memory the hard way: - close session and start new session + run code below 
# ae.load_full_features_from_npz(path_npz = "D:/xc_real_projects/xc_data_history/xc_parus_01/extracted_features/full_features_saec_20250713_232131.npz")
ae.load_full_features_from_npz(path_npz = "D:/xc_real_projects/xc_data_02_Corvidae/extracted_features/full_features_saec_20250713_232131.npz")

ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_reserved())