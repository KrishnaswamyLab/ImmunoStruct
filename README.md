<a id="readme-top"></a>

<!-- PROJECT LOGO -->

<div align="center">

  <h1 align="center">ImmunoStruct</h1>

<!-- PROJECT SHIELDS -->
[![bioRxiv](https://img.shields.io/badge/bioRxiv-ImmunoStruct-firebrick)](https://www.biorxiv.org/content/10.1101/2024.11.01.621580)
[![Twitter Follow](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social)](https://twitter.com/KrishnaswamyLab)
[![GitHub Stars](https://img.shields.io/github/stars/KrishnaswamyLab/ImmunoStruct.svg?style=social\&label=Stars)](https://github.com/KrishnaswamyLab/ImmunoStruct)

  <p align="center">
    A multimodal neural network framework for immunogenicity prediction from peptide-MHC sequence, structure, and biochemical properties
    <br />
    <a href="https://www.biorxiv.org/content/10.1101/2024.11.01.621580"><strong>Explore the paper »</strong></a>
    <br />
    <br />
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<div align="center">
  <img src="assets/schematic.png" alt="ImmunoStruct Architecture" width="800">
</div>

ImmunoStruct is a deep learning framework that integrates sequence, structural, and biochemical information to predict multi-allele class-I peptide-MHC immunogenicity. By leveraging multimodal data from ~27,000 peptide-MHCs, ImmunoStruct significantly improves immunogenicity prediction performance for both infectious disease epitopes and cancer neoepitopes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Key Features

* **Multimodal Integration**: Combines protein sequence, structure, and biochemical properties
* **Novel Cancer-Wildtype Contrastive Learning**: Enhances specificity for cancer neoepitope detection  
* **Enhanced Interpretability**: Provides insights into the molecular basis of immunogenicity

<div align="center">
  <img src="assets/contrastive_learning.png" alt="Contrastive Learning Approach" width="800">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get ImmunoStruct up and running locally, follow these steps.

### Pre-requisites

Before installation, ensure you have:
* Python 3.10+
* CUDA-compatible GPU (recommended)
* Conda package manager
* Weights & Biases account for experiment tracking

### Dependencies
- python 3.10
- torch 2.1.2
- dgl
- torch_geometric 2.5.3

### Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/KrishnaswamyLab/ImmunoStruct.git
   cd ImmunoStruct
   ```

2. **Create and activate conda environment**
   ```sh
   conda create --name immuno python=3.10 -c anaconda -c conda-forge
   conda activate immuno
   ```

3. **Install core dependencies**
   ```sh
   conda install cudatoolkit=11.2 wandb pydantic -c conda-forge
   conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda
   ```

4. **Install PyTorch**
   ```sh
   python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Install DGL**
   ```sh
   python -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
   python -m pip install torchdata==0.7.1
   ```

6. **Install PyTorch Geometric and related packages**
   ```sh
   python -m pip install torch-scatter==2.1.2+pt21cu118 torch-sparse==0.6.18+pt21cu118 torch-cluster==1.6.3+pt21cu118 torch-spline-conv==1.2.2+pt21cu118 torch_geometric==2.5.3 numpy==1.26.3 -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
   ```

7. **Install additional packages**
   ```sh
   python -m pip install graphein[extras]
   python -m pip install lifelines
   python -m pip install -U phate
   python -m pip install multiscale-phate
   ```

8. **Set up environment variables (if needed)**
   ```sh
   export LD_LIBRARY_PATH=/path/to/conda/envs/immuno/lib:$LD_LIBRARY_PATH
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Data Preparation

Place the following files in the `data/` folder:
- `cedar_data_final_with_mprop1_mprop2_v2.txt`
- `complete_score_Mprops_1_2_smoothed_sasa_v2.txt`
- `HLA_27_seqs.csv`

Additionally, ensure you have these folders:
- `graph_pyg_Cancer`
- `graph_pyg_IEDB`

**Generate PyG graph files:**

These PyG graph files can be generated using the below command from the corresponding AlphaFold folders.
```sh
python immunostruct/preprocessing/cancer_graph_construction_new_KBG.py
```


**Compute the biochemical meta-properties (`Mprop1` / `Mprop2`):**

`Mprop1` and `Mprop2` are the 2-d property vector each model consumes. The
tables in `data/` already carry them; this step computes them for a new dataset
or a new set of peptides.

```sh
pip install peptides==0.5.0 mdtraj
```

One table is processed per run, with `--dataset` selecting which column spec to
apply:

```sh
python immunostruct/preprocessing/biochem_properties.py \
    --dataset cedar_wt \
    --in-table  my_peptides.csv \
    --out-table my_peptides_with_mprop.csv
```


`--out-table` is always a new file; input tables are never written to. Add
`--reuse-existing-props` to keep a table's own descriptor columns instead of
recomputing them from sequence, and `--scaler-out` to record the fitted min/max
ranges as JSON.

**The pipeline.** Five stages, from peptide sequence and pMHC structure to the
2-D vector the models consume via `nn.Linear(2, 32)`:

1. **84 biochemical properties** per peptide from the `peptides` package — the 75
   descriptor-scale columns listed in `DESCRIPTORS_75` plus 9 scalars. The
   package version is pinned because `.descriptors()` returns 102 keys in 0.5.0,
   so the original 75 are selected explicitly rather than taken from whatever
   the call returns.
2. **SASA** via `mdtraj.shrake_rupley` (`probe_radius=0.14`,
   `n_sphere_points=960`). The structures are a single fused chain of 272 MHC residues followed by the 9–11mer, so the default
   `--sasa-mode peptide_tail` sums over the trailing peptide residues;
3. **Structure join.** Files are named
   `rank_1_prediction_Immuno<HLA><peptide>_<5hex>.pdb`, where the key is
   `sha1(hla_seq + peptide)[:5]`.
4. **Foreignness**, then **smoothing.** The foreignness score is computed by
   `foreignness.py` (see below) or read from the table, then passed through

5. **Meta-properties.** Each of `Mprop1`, `Mprop2` and `master_property_score`
   is the mean of a min-max-scaled subset of the columns above. The per-dataset
   member lists are in `SPECS` and are fixed inputs.


**Foreignness**

`Mprop1` (IEDB) and `Mprop2` (cedar) both average in `smoothed_foreign`, so a
new peptide set needs a foreignness score before it can have meta-properties at
all. `immunostruct/preprocessing/foreignness.py` computes it, implementing the
multistate thermodynamic model of Luksza et al. 2017:

```
Z(s) = sum_e exp( -k * (a - |s,e|) )        a = 26,  k = 4.86936
R(s) = Z / (1 + Z)
```

where `|s,e|` is the best local alignment score (BLOSUM62, affine gaps, open 11
extend 1) between the peptide and an IEDB immunogenic epitope `e`. The
parameters and the alignment settings follow
[antigen.garnish](https://github.com/andrewrech/antigen.garnish), which produced
the released `Foreignness_Score` column.



```sh
# NCBI BLAST+ is required for the default mode
conda install -c bioconda blast

curl -fsSL https://s3.amazonaws.com/get.rech.io/antigen.garnish-2.3.0.tar.gz -o ag.tar.gz

tar -xzf ag.tar.gz -C data --strip-components=1 --wildcards '*iedb*'   # GNU
```

That yields `data/iedb.fasta` and `data/iedb.bdb.*` (human) plus
`data/Mu_iedb.fasta` (mouse). 

Then either compute the column as part of the Mprop run:

```sh
python immunostruct/preprocessing/biochem_properties.py \
    --dataset iedb --compute-foreignness \
    --in-table  my_peptides.csv \
    --out-table my_peptides_with_mprop.csv
```

or score peptides on their own:

```sh
python immunostruct/preprocessing/foreignness.py \
    --peptides my_peptides.csv --peptide-col peptide \
    --db data/iedb.bdb -o foreignness.csv
```

**Structures**

The SASA stage reads pMHC structures from the folder given to `--pdb-dir`, in
the same way the graph construction step above reads its AlphaFold folders.
Place them under `data/`, for example `data/alphafold_pdb_cedar/`.

Each row is matched to a file by the 5-hex suffix of its filename, which is
`sha1(hla_seq + peptide)[:5]` -- the same key the graph pipeline uses. AlphaFold
output named `rank_1_prediction_Immuno<HLA><peptide>_<5hex>.pdb` therefore joins
without renaming. Rows with no matching structure get `NaN`, and the run reports
how many.

The key comes from the table's `file_id_code` or `id` column when present.

**Putting a new cohort on an existing scale**

Two stages fit on the rows present: the `QuantileTransformer` over the
foreignness column, and the `MinMaxScaler` over each meta-property's inputs. A
cohort processed on its own therefore lands on its own scale. To place it on the
scale of one of the tables in `data/`, append it to that table and process the
two together, keeping the reference rows first and in their original order:

```python
import pandas as pd
ref = pd.read_table("data/cedar_data_final_with_mprop1_mprop2_v2.txt")
new = pd.read_table("new_cohort.txt")
pd.concat([ref, new], ignore_index=True).to_csv(
    "combined.txt", sep="\t", index=False)
```

Process `combined.txt` with the matching `--dataset`, then keep the trailing
rows.

### Training and Testing

1. **Set up Weights & Biases**
   
   Create a project on [Weights & Biases](https://wandb.ai/home) matching your project name.

2. **Run Experiments**
   ```sh
   # HybridModelv2 with full sequence and sequence loss
   python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model HybridModelv2 --wandb-username YOUR_WANDB_USERNAME
   
   # HybridModel with full sequence and sequence loss
   python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model HybridModel --wandb-username YOUR_WANDB_USERNAME
   
   # Sequence with fingerprint model
   python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model SequenceFpModel --wandb-username YOUR_WANDB_USERNAME
   
   # Sequence-only model
   python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --sequence-loss --model SequenceModel --wandb-username YOUR_WANDB_USERNAME
   
   # Structure-only model
   python train_PropIEDB_PropCancer_ImmunoCancer.py --full-sequence --model StructureModel --wandb-username YOUR_WANDB_USERNAME
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- TROUBLESHOOTING -->
## Troubleshooting

### Common Issues

**GLIBCXX Error**
```
ImportError: $some_path/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found
```
**Solution:** Add your conda environment path to `LD_LIBRARY_PATH`:
```sh
export LD_LIBRARY_PATH=/path/to/conda/envs/immuno/lib:$LD_LIBRARY_PATH
```

**CUDA Compatibility Issues**
- Ensure your CUDA version matches the PyTorch installation
- Verify GPU availability with `torch.cuda.is_available()`

**Memory Issues**
- Reduce batch size in training scripts
- Use gradient checkpointing for large models

**Wandb Authentication**
- Login to Wandb: `wandb login`
- Ensure project names match between script and Wandb dashboard

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the Yale License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Krishnaswamy Lab - [@KrishnaswamyLab](https://twitter.com/KrishnaswamyLab)

Project Link: [https://github.com/KrishnaswamyLab/ImmunoStruct](https://github.com/KrishnaswamyLab/ImmunoStruct)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITATION -->
## Citation

If you use ImmunoStruct in your research, please cite our paper:

```bibtex
@article{givechian2024immunostruct,
  title={ImmunoStruct: Integration of protein sequence, structure, and biochemical properties for immunogenicity prediction and interpretation},
  author={Givechian, Kevin Bijan and Rocha, Joao Felipe and Yang, Edward and Liu, Chen and Greene, Kerrie and Ying, Rex and Caron, Etienne and Iwasaki, Akiko and Krishnaswamy, Smita},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[biorxiv-shield]: https://img.shields.io/badge/bioRxiv-ImmunoStruct-firebrick?style=for-the-badge
[biorxiv-url]: https://www.biorxiv.org/content/10.1101/2024.11.01.621580
[twitter-shield]: https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=for-the-badge&logo=twitter&colorB=1DA1F2
[twitter-url]: https://twitter.com/KrishnaswamyLab
[stars-shield]: https://img.shields.io/github/stars/KrishnaswamyLab/ImmunoStruct.svg?style=for-the-badge
[stars-url]: https://github.com/KrishnaswamyLab/ImmunoStruct/stargazers
[issues-shield]: https://img.shields.io/github/issues/KrishnaswamyLab/ImmunoStruct.svg?style=for-the-badge
[issues-url]: https://github.com/KrishnaswamyLab/ImmunoStruct/issues
[license-shield]: https://img.shields.io/badge/license-Yale-blue.svg?style=for-the-badge
[license-url]: https://github.com/KrishnaswamyLab/ImmunoStruct/blob/master/LICENSE.txt
[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[PyG]: https://img.shields.io/badge/PyTorch_Geometric-3C2179?style=for-the-badge&logo=pytorch&logoColor=white
[PyG-url]: https://pytorch-geometric.readthedocs.io/
[DGL]: https://img.shields.io/badge/DGL-FF6B35?style=for-the-badge&logo=python&logoColor=white
[DGL-url]: https://www.dgl.ai/
[Wandb]: https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white
[Wandb-url]: https://wandb.ai/
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
[Conda]: https://img.shields.io/badge/Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white
[Conda-url]: https://conda.io/