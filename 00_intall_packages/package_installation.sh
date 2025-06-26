

pip list

pip uninstall xeno_canto_organizer
pip uninstall fe_idnn
pip uninstall fe_saec

pip install --upgrade https://github.com/sergezaugg/xeno_canto_organizer/releases/download/v0.9.15/xeno_canto_organizer-0.9.15-py3-none-any.whl
pip install --upgrade https://github.com/sergezaugg/feature_extraction_idnn/releases/download/v0.9.5/fe_idnn-0.9.5-py3-none-any.whl
pip install --upgrade https://github.com/sergezaugg/feature_extraction_saec/releases/download/v0.9.5/fe_saec-0.9.5-py3-none-any.whl

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126





