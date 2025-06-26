
# xeno_canto_organizer
pip uninstall xeno_canto_organizer

pip install --upgrade https://github.com/sergezaugg/xeno_canto_organizer/releases/download/v0.9.15/xeno_canto_organizer-0.9.15-py3-none-any.whl

# fe_idnn and fe_saec
pip uninstall fe_idnn
pip uninstall fe_saec

pip install --upgrade https://github.com/sergezaugg/feature_extraction_saec/releases/download/v0.9.6/fe_saec-0.9.6-py3-none-any.whl
pip install --upgrade https://github.com/sergezaugg/feature_extraction_idnn/releases/download/v0.9.6/fe_idnn-0.9.6-py3-none-any.whl

# torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126





