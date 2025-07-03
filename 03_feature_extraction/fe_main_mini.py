#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import torch
from fe_saec import SAEC_extractor
from fe_idnn import IDNN_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# image_path = "D:/xc_real_projects/xc_parus_01/xc_spectrograms"
image_path = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"

# SAEC
path_model = "D:/xc_real_projects/pytorch_hot_models_keep/20250617_150956_encoder_script_GenC_new_TP32_epo007.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
# extract 
ae.extract(image_path = image_path, fe_save_path = "C:/Users/sezau/Downloads", batch_size = 32, shuffle = True , n_batches = 10) 
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# IDNN 
ie = IDNN_extractor(model_tag = "ResNet50")
ie.eval_nodes
ie.create("layer1.2.conv3")
# extract 
ie.extract(image_path = image_path, fe_save_path = "C:/Users/sezau/Downloads",  freq_pool = 4, batch_size = 16, n_batches = 100, ecut = 1)
[ie.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]































