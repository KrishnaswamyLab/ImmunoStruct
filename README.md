<h1 align="center">
ImmunoStruct
</h1>

<p align="left">
<strong>A multimodal neural network framework for immunogenicity prediction from peptide-MHC sequence, structure, and biochemical properties</strong>
</p>

<div align="center">

[![bioRxiv](https://img.shields.io/badge/bioRxiv-ImmunoStruct-firebrick)](https://www.biorxiv.org/content/10.1101/2024.11.01.621580)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/KrishnaswamyLab/ImmunoStruct.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/ImmunoStruct/)

</div>


## Citation
```
@article{givechian2024immunostruct,
  title={ImmunoStruct: Integration of protein sequence, structure, and biochemical properties for immunogenicity prediction and interpretation},
  author={Givechian, Kevin Bijan and Rocha, Joao Felipe and Yang, Edward and Liu, Chen and Greene, Kerrie and Ying, Rex and Caron, Etienne and Iwasaki, Akiko and Krishnaswamy, Smita},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Schematic
<img src = "assets/schematic.png" width=800>

## A novel cancer-wildtype contrastive learning
<img src = "assets/contrastive_learning.png" width=800>


## Abstract
Epitope-based vaccines are promising therapeutic modalities for infectious diseases and cancer, but identifying immunogenic epitopes is challenging. The vast majority of prediction methods only use amino acid sequence information, and do not incorporate wide-scale structure data and biochemical properties across each peptide-MHC. We present ImmunoStruct, a deep-learning model that integrates sequence, structural, and biochemical information to predict multi-allele class-I peptide-MHC immunogenicity. By leveraging a multimodal dataset of $\sim$27,000 peptide-MHCs, we demonstrate that ImmunoStruct improves immunogenicity prediction performance and interpretability beyond existing methods, across infectious disease epitopes and cancer neoepitopes. We further show strong alignment with \textit{in vitro} assay results for a set of SARS-CoV-2 epitopes, as well as strong performance in peptide-MHC-based cancer patient survival prediction. Overall, this work also presents a new architecture that incorporates equivariant graph processing and multimodal data integration for the long standing task in immunotherapy.


## Usage
### Data preparation
Under the `data/` folder, you should have the following files.
- `cedar_data_final_with_mprop1_mprop2_v2.txt`
- `complete_score_Mprops_1_2_smoothed_sasa_v2.txt`
- `HLA_27_seqs_csv.csv`

Besides, you should have the following folders.
- `graph_pyg_Cancer`
- `graph_pyg_IEDB`

These PyG graph files can be generated using `immunostruct/preprocessing/cancer_graph_construction_new_KBG.py` from the corresponding AlphaFold folders.

### Training and testing
You first need to login to your wandb account on the website: https://wandb.ai/home, and create a project with the name matching the wandb project string in the corresponding training file (look for `wandb.init`).

Then, you can run the training scripts with your wandb username (`YOUR_WANDB_USERNAME`).

#### Experiments on IEDB infectious diseases.
TO BE UPDATED!!!!
```
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model HybridModelv2 --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model HybridModel --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model SequenceFpModel --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model SequenceModel --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --model StructureModel --wandb-username $YOUR_WANDB_USERNAME
```

#### Experiments on human cancer neoepitopes.
TO BE UPDATED!!!!


## Environment
```
conda create --name immuno python=3.10 -c anaconda -c conda-forge
conda activate immuno
conda install cudatoolkit=11.2 wandb pydantic -c conda-forge
conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
python -m pip install torchdata==0.7.1
python -m pip install torch-scatter==2.1.2+pt21cu118 torch-sparse==0.6.18+pt21cu118 torch-cluster==1.6.3+pt21cu118 torch-spline-conv==1.2.2+pt21cu118 torch_geometric==2.5.3 numpy==1.26.3 -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
python -m pip install graphein[extras]
python -m pip install lifelines

python -m pip install -U phate
python -m pip install multiscale-phate
```


## Debug
1. `ImportError: $some_path/libstdc++.so.6: version `GLIBCXX_3.4.29' not found.`
    You can try adding your immuno conda environment path to `LD_LIBRARY_PATH`.
    ```
    export LD_LIBRARY_PATH=$PATH_TO_CONDA_ENV:$LD_LIBRARY_PATH
    ```
    In my case, it would be:
    ```
    export LD_LIBRARY_PATH=/home/cl2482/.conda/envs/immuno/lib:$LD_LIBRARY_PATH
    ```
