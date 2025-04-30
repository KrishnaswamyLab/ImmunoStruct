# ImmunoStruct

## Overview
Within the therapeutic landscape of vaccine-based immunotherapy, epitope based vaccines have emerged as an exciting therapeutic modality. However, epitope-based vaccine success hinges on the precise prioritization of class-I neoantigens that are immunogenic - an attribute observed in only a fraction of the total peptides used to vaccinate the patient. Current prioritization methods are primarily sequence-based not accounting for the intricate interplay between the structural and sequential characteristics of the peptide-MHC complex (pMHC). The clinical implications of this shortcoming are significant given the necessity to restrict encapsulated epitope delivery to a selected number of candidates - often restricted to less than several dozen. Therefore, it is of high clinical interest to increase the percentage of immunogenic administered peptides. The present study explores the wide scale integration of structural and sequence data to enhance the accuracy and interpretability of immunogenicity prediction. To this end, we have curated a dataset of ~27,000 peptide-MHC complexes using AlphaFold2, with each complex paired to an experimentally determined immunogenicity measurement. By leveraging this data, we developed \textit{ImmunoStruct}, a deep-learning model that integrates both sequence and structural information to predict peptide-MHC immunogenicity. We further identify potential relationships between different structural features of the peptide-MHC complex to interpret physicochemical properties underpinning immunogenicity and explore substructural pMHC immunogenicity motifs. This work seeks to improve epitope-based vaccine design by integrating structural information with pMHC sequence information to improve the prediction of immunogenic epitopes.

## Graphical Abstract

## Citation

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
```
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model HybridModelv2 --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model HybridModel --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model SequenceFpModel --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model SequenceModel --wandb-username $YOUR_WANDB_USERNAME
python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --model StructureModel --wandb-username $YOUR_WANDB_USERNAME
```

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
# ImmunoStruct
