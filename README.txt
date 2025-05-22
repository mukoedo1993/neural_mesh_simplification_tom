It is Tom (Zichun Wang)'s customization on https://github.com/martinnormark/neural-mesh-simplification/

Step 1: In examples/data folder, put your own .obj files. I put files from https://github.com/ranahanocka/MeshCNN/
Step 2: 
### Installation
```bash
conda create -n neural-mesh-simplification python=3.12
conda activate neural-mesh-simplification
conda install pip
```

Step 3:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_cluster==1.6.3 torch_geometric==2.5.3 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

Step 4: 
```bash
pip install -r requirements.txt
pip install -e .
```

Step 5:
Now, you have processed data in examples/data/processed folder.
