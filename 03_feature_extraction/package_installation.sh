# First install torch and all its deps
pip install torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

#  fe_saec
pip uninstall fe_saec
pip install --upgrade https://github.com/sergezaugg/feature_extraction_saec/releases/download/v0.9.13/fe_saec-0.9.13-py3-none-any.whl

# fe_idnn 
pip uninstall fe_idnn
pip install --upgrade https://github.com/sergezaugg/feature_extraction_idnn/releases/download/v0.9.12/fe_idnn-0.9.12-py3-none-any.whl


