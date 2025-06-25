#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

# pip install --upgrade https://github.com/sergezaugg/feature_extraction_idnn/releases/download/v0.9.5/fe_idnn-0.9.5-py3-none-any.whl
# pip install --upgrade https://github.com/sergezaugg/feature_extraction_saec/releases/download/v0.9.5/fe_saec-0.9.5-py3-none-any.whl

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126

import torch
from fe_saec import SAEC_extractor
from fe_idnn import IDNN_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_path = "D:/xc_real_projects/xc_parus_01/xc_spectrograms"
image_path = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"



# SAEC
path_model = "D:/xc_real_projects/pytorch_hot_models_keep/20250617_150956_encoder_script_GenC_new_TP32_epo007.pth"
ae = SAEC_extractor(path_model = path_model, device = device) 
# extract 
ae.extract(image_path = image_path, batch_size = 32, shuffle = True , devel = True) 
ae.time_pool(ecut=2)
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]  

# IDNN 
ie = IDNN_extractor(model_tag = "ResNet50")
ie.eval_nodes
ie.create("layer1.2.conv3")
# extract 
ie.extract(image_path = image_path, freq_pool = 4, batch_size = 16, n_batches = 10, ecut = 1)
[ie.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]




















#-----------------------------------
# main process starts here 
print("Activating session ...")
import torch
from fe_idnn.tools import FeatureExtractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


n_batches = 10
image_path = "D:/xc_real_projects/xc_parus_01/xc_spectrograms"


fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer1.2.conv3")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer2.3.conv3")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer3.5.conv3")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer4.2.conv3")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 0)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = FeatureExtractor(model_tag = "DenseNet121")
fe.eval_nodes
fe.create("features.denseblock3")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = FeatureExtractor(model_tag = "vgg16")
fe.eval_nodes
fe.create("features.28")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = FeatureExtractor(model_tag = "MaxVit_T")
fe.eval_nodes
fe.create("blocks.3.layers.1.layers.MBconv.layers.conv_c")
fe.extract(image_path, freq_pool = 1, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]













