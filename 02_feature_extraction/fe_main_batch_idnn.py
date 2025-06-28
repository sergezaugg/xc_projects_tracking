#--------------------             
# Author : Serge Zaugg
# Description : Batch feature extraction form IDNN models with constant frequency pooling
#--------------------

import os
import torch
from fe_idnn import IDNN_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path_source_images = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
# path_save_features = "C:/Users/sezau/Downloads/xc_sw_europe"

# path_source_images = "D:/xc_real_projects/xc_parus_01/xc_spectrograms"
# path_save_features = "C:/Users/sezau/Downloads/xc_parus_01"

path_source_images = "D:/xc_real_projects/xc_corvus_corax/xc_spectrograms"
path_save_features = "C:/Users/sezau/Downloads/xc_corvus_corax"

if not os.path.exists(path_save_features):
    os.mkdir(path_save_features) 

n_batches = 5
# n_batches = 1000000000

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





