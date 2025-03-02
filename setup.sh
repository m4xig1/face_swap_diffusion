# install correct version of torch and torchvision according to your cuda version
# CHANGED TO 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121    
pip install -r requirements.txt
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
cd eval_tool/Deep3DFaceRecon_pytorch_edit/nvdiffrast/
git clone https://github.com/NVlabs/nvdiffrast
pip install -e .
