<a id="readme-top"></a>

<!-- PROJECT LOGO -->

<div align="center">
  <h1><img src="assets/ImmunoStruct_cover.png" width="200"><br><code>ImmunoStruct</code></h1>
  <h3>ImmunoStruct enables multimodal deep learning for immunogenicity prediction</h3>

  [![nature](https://img.shields.io/badge/nature-machine_intelligence-gold)](https://www.nature.com/articles/s42256-025-01163-y)
  [![PDF](https://img.shields.io/badge/PDF-DADBDD)](https://www.nature.com/articles/s42256-025-01163-y.pdf)
  [![Huggingface](https://img.shields.io/badge/Dataset-ImmunoStruct-orange)](https://huggingface.co/datasets/ChenLiu1996/ImmunoStruct)
  [![Huggingface](https://img.shields.io/badge/Model-ImmunoStruct-orange)](https://huggingface.co/ChenLiu1996/ImmunoStruct)
  [![GitHub Stars](https://img.shields.io/github/stars/KrishnaswamyLab/ImmunoStruct.svg?style=social\&label=Stars)](https://github.com/KrishnaswamyLab/ImmunoStruct)
  <br>[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kevin-blue)](https://www.linkedin.com/in/kevin-bijan-givechian-phd-36467ba3/)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-João-blue)](https://www.linkedin.com/in/joao-felipe-rocha/)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Chen-blue)](https://www.linkedin.com/in/chenliu1996/)
  [![Google Scholar](https://img.shields.io/badge/Google_Scholar-Chen-4a86cf?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=3rDjnykAAAAJ&sortby=pubdate)
  <br>[![Twitter Follow](https://img.shields.io/twitter/follow/Kevin.svg?style=social)](https://x.com/KevinGivechian)
  [![Twitter Follow](https://img.shields.io/twitter/follow/Chen.svg?style=social)](https://x.com/ChenLiu_1996)
  [![Twitter Follow](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social)](https://x.com/KrishnaswamyLab)

</div>

Project leads: [Kevin Bijan Givechian](https://www.linkedin.com/in/kevin-bijan-givechian-phd-36467ba3/), [João Felipe Rocha](https://www.linkedin.com/in/joao-felipe-rocha/), [Chen Liu](https://www.linkedin.com/in/chenliu1996/).
<br>Correspondence: `akiko.iwasaki@yale.edu`, `smita.krishnaswamy@yale.edu`.

<br>In case you don't have access to Nature, here are the [main paper](pdf/ImmunoStruct_NMI.pdf) and the [supplementary materials](pdf/ImmunoStruct_NMI_supp.pdf).

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#news">News</a></li>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## News

News in English<br>
[![news](https://img.shields.io/badge/news-Yale_School_of_Medicine-0F4D92)](https://medicine.yale.edu/news-article/using-machine-learning-to-develop-personalized-vaccines-for-cancer/)
[![news](https://img.shields.io/badge/news-Time_News_(Mar_5)-white)](https://time.news/machine-learning-immune-system-personalized-medicine-clues/)
[![news](https://img.shields.io/badge/news-Time_News_(Mar_2)-white)](https://time.news/ai-improves-personalized-cancer-vaccine-design-yale-study/)
[![news](https://img.shields.io/badge/news-Innovita_Research-white)](https://www.innovitaresearch.com/2026/03/04/using-machine-learning-to-develop-personalized-vaccines-for-cancer/)
[![news](https://img.shields.io/badge/news-Medical_Xpress-white)](https://medicalxpress.com/news/2026-02-personalized-vaccines-cancer-machine.html)
[![news](https://img.shields.io/badge/news-Technology.org-white)](https://www.technology.org/2026/03/04/using-machine-learning-to-develop-personalized-vaccines-for-cancer/)
[![news](https://img.shields.io/badge/news-Decoding_Bio-white)](https://decodingbio.substack.com/p/biobyte-144-a-virtual-model-of-cell)
[![news](https://img.shields.io/badge/news-Bioengineer-white)](https://bioengineer.org/immunostruct-advancing-deep-learning-in-immunogenicity-prediction/)
[![news](https://img.shields.io/badge/news-GeneOnline-white)](https://www.geneonline.com/deep-learning-model-immunostruct-developed-to-improve-prediction-of-immunogenic-epitopes-for-vaccine-research/)
<br>[![wiki](https://img.shields.io/badge/wiki-Wikipedia_Immunogenicity-pink)](https://en.wikipedia.org/wiki/Immunogenicity)


News in Chinese<br>
[![news](https://img.shields.io/badge/news-新智元@网易-gold)](https://www.163.com/dy/article/KJ13HJ340511ABV6.html)
[![news](https://img.shields.io/badge/news-腾讯云_(Tencent_Cloud)-white)](https://cloud.tencent.com/developer/article/2612794)
<br>[![news](https://img.shields.io/badge/news-新智元@搜狐-F6E7B2)](https://m.sohu.com/a/974949014_473283)
[![news](https://img.shields.io/badge/news-新智元@智源社区-F6E7B2)](https://hub.baai.ac.cn/view/51813)
[![news](https://img.shields.io/badge/news-新浪财经_转发_新智元-F6E7B2)](https://finance.sina.com.cn/stock/t/2026-01-11/doc-inhfxvtm9135944.shtml)
[![news](https://img.shields.io/badge/news-AI中文社_转发_新智元-F6E7B2)](https://www.aizws.net/news/detail/6922)
[![news](https://img.shields.io/badge/news-百度_转发_新智元-F6E7B2)](https://baijiahao.baidu.com/s?id=1854022994702774585)
<br>[![wiki](https://img.shields.io/badge/wiki-百度百科_免疫原性-pink)](https://baike.baidu.com/item/%E5%85%8D%E7%96%AB%E5%8E%9F%E6%80%A7/5292060)
[![wiki](https://img.shields.io/badge/wiki-百度百科_图神经网络-pink)](https://baike.baidu.com/item/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/59091829)


&#9744; TODO: create and release an end-to-end tool.
<br>&#x2705; Feb 20, 2026: The [multimodal datasets](https://huggingface.co/datasets/ChenLiu1996/ImmunoStruct) and [model weights](https://huggingface.co/ChenLiu1996/ImmunoStruct) are now open-sourced. See [instructions](https://github.com/KrishnaswamyLab/ImmunoStruct?tab=readme-ov-file#data-preparation).
<br>&#x2705; Dec 31, 2025: **[Published](https://www.nature.com/articles/s42256-025-01163-y) in Nature Machine Intelligence.**
<br>&#x2705; Dec 04, 2025: Informally presented at NeurIPS 2025 (did not submit, no dual-submission concern).
<br>&#x2705; Aug 18, 2025: Received the [Colton Innovation Fund](https://ventures.yale.edu/news/yales-colton-center-autoimmunity-announces-2025-awardees-advancing-innovation-autoimmune) from [Colton Center for Autoimmunity at Yale University](https://ventures.yale.edu/colton-center-for-autoimmunity).
<br>&#x2705; May 06, 2025: Submitted to Nature Machine Intelligence.
<br>&#x2705; Nov 05, 2024: Presented at MoML@MIT 2024 (non-archival abstract & poster).
<br>&#x2705; Nov 01, 2024: [Preprint](https://www.biorxiv.org/content/10.1101/2024.11.01.621580) released.

<!-- ABOUT THE PROJECT -->
## About The Project

<div align="center">
  <img src="assets/schematic.png" alt="ImmunoStruct Architecture" width="800">
</div>

ImmunoStruct is a multimodal deep learning framework that integrates sequence, structural, and biochemical information to predict multi-allele class-I peptide-MHC immunogenicity. By leveraging multimodal data from 26,049 peptide-MHCs and jointly modeling sequence and structure, ImmunoStruct significantly improves immunogenicity prediction performance for both infectious disease epitopes and cancer neoepitopes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Key Features

* **Multimodal Integration**: Combines peptide-MHC protein sequence, structure, and biochemical properties
* **Novel Cancer-Wildtype Contrastive Learning**: Enhances specificity for cancer neoepitope detection
* **Enhanced Interpretability**: Provides insights into the substructural basis of immunogenicity

<div align="center">
  <img src="assets/contrastive_learning.png" alt="Contrastive Learning Approach" width="800">
  <img src="assets/visualizations.png" alt="Visualizations" width="800">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITATION -->
## Citation

If you use ImmunoStruct in your research, please cite our paper:

BibTeX:
```bibtex
@article{givechian2026immunostruct,
  title={ImmunoStruct enables multimodal deep learning for immunogenicity prediction},
  author={Givechian, Kevin Bijan and Rocha, Jo{\~a}o Felipe and Liu, Chen and Yang, Edward and Tyagi, Sidharth and Greene, Kerrie and Ying, Rex and Caron, Etienne and Iwasaki, Akiko and Krishnaswamy, Smita},
  journal={Nature Machine Intelligence},
  volume={8},
  pages={70--83},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```
Nature format:<br>
Givechian, K.B., Rocha, J.F., Liu, C. et al. ImmunoStruct enables multimodal deep learning for immunogenicity prediction. *Nat Mach Intell* 8, 70–83 (2026). https://doi.org/10.1038/s42256-025-01163-y


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get ImmunoStruct up and running locally, follow these steps.

### Pre-requisites

Before installation, ensure you have:
* CUDA-compatible GPU (recommended)
* Conda package manager
* Weights & Biases account for experiment tracking

### Installation

1. **Clone the repository**
    ```sh
    git clone https://github.com/KrishnaswamyLab/ImmunoStruct.git
    cd ImmunoStruct
    ```

2. **Create conda environment and install dependencies**
    ```sh
    conda create --name immuno python=3.8 -c anaconda -c conda-forge -y
    conda activate immuno
    conda install cudatoolkit=11.2 wandb pydantic -c conda-forge -y
    conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda -y
    python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    python -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
    python -m pip install torchdata==0.7.1
    python -m pip install torch-scatter==2.1.2+pt21cu118 torch-sparse==0.6.18+pt21cu118 torch-cluster==1.6.3+pt21cu118 torch-spline-conv==1.2.2+pt21cu118 torch_geometric==2.5.3 numpy==1.21.1 -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
    python -m pip install jax==0.2.25 jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    python -m pip install "alphafold-colabfold==2.0.0" "colabfold==1.2.0" "dm-haiku==0.0.4"
    python -m pip install "biopython==1.78"
    python -m pip install graphein[extras]
    python -m pip install lifelines
    python -m pip install huggingface_hub
    python -m pip install ipykernel
    python -m pip install ipywidgets
    ```

    The following steps might be necessary if you encounter problems when running the inference. These are some package incompatibilities that we managed to resolve in a manual way:
    - Go to `PATH_TO_ENV/lib/python3.8/site-packages/jaxlib/xla_client.py`: change `np.object` to `object`.
    - Go to `PATH_TO_ENV/lib/python3.8/site-packages/alphafold/common/residue_constants.py`: change `np.int` to `np.int32`.
    - Go to `PATH_TO_ENV/lib/python3.8/site-packages/alphafold/data/templates.py`: change `np.object` to `object`.

3. Create and build another environment for obtaining MSAs locally. Only relevant if you want to run your own protein folding.
    ```sh
    conda create --name local_msa python=3.10 -c anaconda -c conda-forge -y
    conda activate local_msa
    conda install cudatoolkit=11.2 wandb pydantic -c conda-forge -y
    conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda -y
    python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install colabfold[alphafold]==1.5.5
    pip install jax==0.4.23 jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    conda install -c bioconda mmseqs2 -y
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
1. Download the dataset from huggingface. You need to sign up for huggingface when prompted. Starting at the project root (`ImmunoStruct`):
    ```sh
    conda activate immuno
    cd ./data/
    hf download ChenLiu1996/ImmunoStruct --repo-type dataset --local-dir ./
    ```

2. Download the pre-trained model weights from huggingface. Starting at the project root (`ImmunoStruct`):
    ```sh
    cd ./checkpoints/
    hf download ChenLiu1996/ImmunoStruct --local-dir ./
    ```

3. Make sure the following files are in the `data` folder:
    - `ImmunoStruct_IEDB_data.csv`
    - `ImmunoStruct_CEDAR_data_cancer.csv`
    - `ImmunoStruct_CEDAR_data_wildtype.csv`
    - `ImmunoStruct_clinical_data.csv`
    - `ImmunoStruct_clinical_data_survival.csv`
    - `HLA_allele_sequences.csv`

4. Unzip the graph structure PyTorch files.
    ```sh
    unzip graph_pyg_IEDB.zip
    unzip graph_pyg_CEDAR_cancer.zip
    unzip graph_pyg_CEDAR_wildtype.zip
    unzip graph_pyg_clinical.zip
    ```

   Now the following folders should be under the `data` folder:
    - `graph_pyg_IEDB`
    - `graph_pyg_CEDAR_cancer`
    - `graph_pyg_CEDAR_wildtype`
    - `graph_pyg_clinical`

5. If you want to customize the graph-building logic, the graph structure PDB files produced by AlphaFold2 are already made available by the same huggingface download command. Unzip the corresponding zip files and you will have the following folders.
    - `alphafold2_pdb_IEDB`
    - `alphafold2_pdb_CEDAR_cancer`
    - `alphafold2_pdb_CEDAR_wildtype`
    - `alphafold2_pdb_clinical`

### AlphaFold2 Structure Data

We have provided the structure data encoded as PyTorch Geometric (PyG) graphs on huggingface. You just need to follow the instruction in the previous **Data Preparation** section. You can skip this section if you are not planning to fold your own data.

<details>
  <summary>How the PyG graphs are generated</summary>

<br>The PyG graphs are generated using the following steps. The generation scripts are available in `immunostruct/preprocessing`, in case you ever need to run some or all of them.

- Option 1 is easy to perform, but it's slow and rate-limited.
- Option 2 involves more steps, but it is more suitable to larger datasets (>2000 sequences).

1. Option 1: Using the online MSA server (slow, rate-limited, not recommended for >2000 sequences). Starting at `ImmunoStruct` root folder.
    ```sh
    # [CPU] Step 1-3. Prepare MSA for AlphaFold.
    # Download colabfold and REMEMBER where it is downloaded to. Likely default to `~/.cache/colabfold`.
    python -m colabfold.download

    # Prepare MSA.
    conda activate immuno
    cd immunostruct/preprocessing
    python step1-3_server_sequence_to_msa.py \
        --input-csv ../../data/ImmunoStruct_IEDB_data.csv \
        --output-dir ../../data/pdb_files/IEDB/ \
        --tmp-dir /tmp/ \
        --allele-col-name allele \
        --peptide-col-name peptide

    python step1-3_server_sequence_to_msa.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_cancer.csv \
        --output-dir ../../data/pdb_files/CEDAR_cancer/ \
        --tmp-dir /tmp/ \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    python step1-3_server_sequence_to_msa.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_wildtype.csv \
        --output-dir ../../data/pdb_files/CEDAR_wildtype/ \
        --tmp-dir /tmp/ \
        --allele-col-name allele \
        --peptide-col-name wt_pep

    python step1-3_server_sequence_to_msa.py \
        --input-csv ../../data/ImmunoStruct_clinical_data.csv \
        --output-dir ../../data/pdb_files/clinical/ \
        --tmp-dir /tmp/ \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    # [GPU] Step 4. AlphaFold2.
    # start/end help run multiple jobs in parallel.
    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_IEDB_data.csv \
        --output-dir ../../data/pdb_files/IEDB/ \
        --start 0 --end 24540 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name peptide

    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_cancer.csv \
        --output-dir ../../data/pdb_files/CEDAR_cancer/ \
        --start 0 --end 2801 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_wildtype.csv \
        --output-dir ../../data/pdb_files/CEDAR_wildtype/ \
        --start 0 --end 2801 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name wt_pep

    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_clinical_data.csv \
        --output-dir ../../data/pdb_files/clinical/ \
        --start 0 --end 29485 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    # [CPU] Step 5. Moving and renaming the structure data in PDB files.
    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/IEDB/ \
        --output-dir ../../data/alphafold2_pdb_IEDB/

    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/CEDAR_cancer/ \
        --output-dir ../../data/alphafold2_pdb_CEDAR_cancer/

    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/CEDAR_wildtype/ \
        --output-dir ../../data/alphafold2_pdb_CEDAR_wildtype/

    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/clinical/ \
        --output-dir ../../data/alphafold2_pdb_clinical/

    # [CPU] Step 6. Generating PyG graphs (structures in PDB files to structures in PyTorch .pt files).
    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_IEDB/ \
        --output-dir ../../data/graph_pyg_IEDB/

    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_CEDAR_cancer/ \
        --output-dir ../../data/graph_pyg_CEDAR_cancer/

    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_CEDAR_wildtype/ \
        --output-dir ../../data/graph_pyg_CEDAR_wildtype/

    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_clinical/ \
        --output-dir ../../data/graph_pyg_clinical/
    ```

2. Option 2: Performing MSA locally (what we did). Starting at `ImmunoStruct` root folder.
    ```sh
    # [CPU] Step 1-3. Prepare MSA for AlphaFold.
    # Download colabfold and REMEMBER where it is downloaded to.
    python -m colabfold.download

    # Download the MSA database locally.
    mkdir ./database_msa/
    cd ./database_msa/
    wget https://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2302.tar.gz
    tar -xzvf uniref30_2302.tar.gz
    mmseqs tsv2exprofiledb uniref30_2302 uniref30_2302_db
    wget https://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz
    tar -xzvf colabfold_envdb_202108.tar.gz
    mmseqs tsv2exprofiledb colabfold_envdb_202108 colabfold_envdb_202108_db
    cd ..

    # Prepare MSA.
    conda activate local_msa
    cd immunostruct/preprocessing
    python step1_local_sequence_to_fasta.py \
        --input-csv ../../data/ImmunoStruct_IEDB_data.csv \
        --output-fasta ../../data/fasta/ImmunoStruct_IEDB_data.fasta \
        --allele-col-name allele \
        --peptide-col-name peptide
    python step2_local_fasta_to_a3m.py \
        --input-fasta ../../data/fasta/ImmunoStruct_IEDB_data.fasta \
        --msa-database-dir ../../database_msa/ \
        --output-dir ../../data/a3m/IEDB/
    python step3_local_a3m_to_msa.py \
        --input-dir ../../data/a3m/IEDB/ \
        --output-dir ../../data/pdb_files/IEDB/ \
        --input-csv ../../data/ImmunoStruct_IEDB_data.csv \
        --allele-col-name allele \
        --peptide-col-name peptide

    python step1_local_sequence_to_fasta.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_cancer.csv \
        --output-fasta ../../data/fasta/ImmunoStruct_CEDAR_data_cancer.fasta \
        --allele-col-name allele \
        --peptide-col-name mut_pep
    python step2_local_fasta_to_a3m.py \
        --input-fasta ../../data/fasta/ImmunoStruct_CEDAR_data_cancer.fasta \
        --msa-database-dir ../../database_msa/ \
        --output-dir ../../data/a3m/CEDAR_cancer/
    python step3_local_a3m_to_msa.py \
        --input-dir ../../data/a3m/CEDAR_cancer/ \
        --output-dir ../../data/pdb_files/CEDAR_cancer/ \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_cancer.csv \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    python step1_local_sequence_to_fasta.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_wildtype.csv \
        --output-fasta ../../data/fasta/ImmunoStruct_CEDAR_data_wildtype.fasta \
        --allele-col-name allele \
        --peptide-col-name wt_pep
    python step2_local_fasta_to_a3m.py \
        --input-fasta ../../data/fasta/ImmunoStruct_CEDAR_data_wildtype.fasta \
        --msa-database-dir ../../database_msa/ \
        --output-dir ../../data/a3m/CEDAR_wildtype/
    python step3_local_a3m_to_msa.py \
        --input-dir ../../data/a3m/CEDAR_wildtype/ \
        --output-dir ../../data/pdb_files/CEDAR_wildtype/ \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_wildtype.csv \
        --allele-col-name allele \
        --peptide-col-name wt_pep

    python step1_local_sequence_to_fasta.py \
        --input-csv ../../data/ImmunoStruct_clinical_data.csv \
        --output-fasta ../../data/fasta/ImmunoStruct_clinical_data.fasta \
        --allele-col-name allele \
        --peptide-col-name mut_pep
    python step2_local_fasta_to_a3m.py \
        --input-fasta ../../data/fasta/ImmunoStruct_clinical_data.fasta \
        --msa-database-dir ../../database_msa/ \
        --output-dir ../../data/a3m/clinical/
    python step3_local_a3m_to_msa.py \
        --input-dir ../../data/a3m/clinical/ \
        --output-dir ../../data/pdb_files/clinical/ \
        --input-csv ../../data/ImmunoStruct_clinical_data.csv \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    # [GPU] Step 4. AlphaFold2.
    # start/end help run multiple jobs in parallel.
    conda deactivate
    conda activate immuno
    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_IEDB_data.csv \
        --output-dir ../../data/pdb_files/IEDB/ \
        --start 0 --end 24540 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name peptide

    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_cancer.csv \
        --output-dir ../../data/pdb_files/CEDAR_cancer/ \
        --start 0 --end 2801 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_CEDAR_data_wildtype.csv \
        --output-dir ../../data/pdb_files/CEDAR_wildtype/ \
        --start 0 --end 2801 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name wt_pep

    python step4_msa_to_pdb.py \
        --input-csv ../../data/ImmunoStruct_clinical_data.csv \
        --output-dir ../../data/pdb_files/clinical/ \
        --start 0 --end 29485 \
        --params-loc /path/to/colabfold \
        --allele-col-name allele \
        --peptide-col-name mut_pep

    # [CPU] Step 5. Moving and renaming the structure data in PDB files.
    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/IEDB/ \
        --output-dir ../../data/alphafold2_pdb_IEDB/

    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/CEDAR_cancer/ \
        --output-dir ../../data/alphafold2_pdb_CEDAR_cancer/

    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/CEDAR_wildtype/ \
        --output-dir ../../data/alphafold2_pdb_CEDAR_wildtype/

    python step5_rename_pdb.py \
        --input-dir ../../data/pdb_files/clinical/ \
        --output-dir ../../data/alphafold2_pdb_clinical/

    # [CPU] Step 6. Generating PyG graphs (structures in PDB files to structures in PyTorch .pt files).
    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_IEDB/ \
        --output-dir ../../data/graph_pyg_IEDB/

    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_CEDAR_cancer/ \
        --output-dir ../../data/graph_pyg_CEDAR_cancer/

    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_CEDAR_wildtype/ \
        --output-dir ../../data/graph_pyg_CEDAR_wildtype/

    python step6_pdb_to_pyg.py \
        --input-dir ../../data/alphafold2_pdb_clinical/ \
        --output-dir ../../data/graph_pyg_clinical/
    ```

</details>
<br>

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
1. **Activate the environment**
    ```sh
    conda activate immuno
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    ```

2. **Set up Weights & Biases**

   Create a project on [Weights & Biases](https://wandb.ai/home) matching your project name.

3. **Our main experiments**

    These are examples for training ImmunoStruct.

    ```sh
    # IEDB training
    python train_IEDB_wFT.py --model HybridModelv2 --sequence-loss --full-sequence --seed 1 --wandb-username YOUR_WANDB_USERNAME

    # CEDAR training
    python train_CEDAR_wFT.py --model HybridModel_Comparative --sequence-loss --full-sequence --comparative --seed 1 --wandb-username YOUR_WANDB_USERNAME
    ```

    For running inference using the models we provide:
    ```sh
    # IEDB inference
    python infer_IEDB_or_CEDAR.py --infer_dataset IEDB --model HybridModelv2 --model-path ../checkpoints/IEDB_model_seed1.pt --full-sequence --seed 1

    # CEDAR inference
    python infer_IEDB_or_CEDAR.py --infer_dataset CEDAR --model HybridModel_Comparative --model-path ../checkpoints/CEDAR_model_seed2.pt --full-sequence --seed 2

    # Clinical inference
    python infer_clinical_only.py --model HybridModel_Comparative --model-path ../checkpoints/CEDAR_model_seed2.pt --full-sequence
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
conda activate immuno
echo $CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
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
