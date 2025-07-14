# First install torch and all its deps
pip install torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# then the packages with minimal deps 
pip uninstall train_saec
pip install --upgrade https://github.com/sergezaugg/train_saec/releases/download/v0.9.7/train_saec-0.9.7-py3-none-any.whl


