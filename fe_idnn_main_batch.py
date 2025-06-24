#--------------------             
# Author : Serge Zaugg
# Description : A python cli tool to keep track and reproduce feature extraction history
# Applies FeatureExtractor with several models and params
# Usage example: python main_batch.py -i "D:/xc_real_projects/xc_sw_europe/xc_spectrograms" -n 21
# spec example:  python main_batch.py -i "D:/xc_real_projects/xc_sw_europe/xc_spectrograms" -n 1000000
# spec example:  python main_batch.py -i "D:/xc_real_projects/xc_parus_01/xc_spectrograms" -n 1000000
# spec example:  python main_batch.py -i "D:/xc_real_projects/xc_corvus_corax/xc_spectrograms" -n 1000000
# Usage example: python main_batch.py -i "./dev_data/images" -n 31
#--------------------

# pip install https://github.com/sergezaugg/feature_extraction_idnn/releases/download/v0.9.2/fe_idnn-0.9.2-py3-none-any.whl
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

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




