
# xeno_canto_organizer
pip uninstall xeno_canto_organizer

pip install --upgrade https://github.com/sergezaugg/xco/releases/download/v0.9.3/xeno_canto_organizer-0.9.3-py3-none-any.whl

# First install torch and all its deps
pip install torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# fe_idnn and fe_saec
pip uninstall train_saec
pip uninstall fe_idnn
pip uninstall fe_saec

# then  the packages with minimal deps 
pip install --upgrade https://github.com/sergezaugg/train_saec/releases/download/v0.9.0/train_saec-0.9.0-py3-none-any.whl
pip install --upgrade https://github.com/sergezaugg/feature_extraction_idnn/releases/download/v0.9.12/fe_idnn-0.9.12-py3-none-any.whl
pip install --upgrade https://github.com/sergezaugg/feature_extraction_saec/releases/download/v0.9.8/fe_saec-0.9.8-py3-none-any.whl


