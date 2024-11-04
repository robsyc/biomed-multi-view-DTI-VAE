# Context

This repository was forked from [IBM's Multi-view Molecular Embedding with Late Fusion (MMELON) architecture](https://github.com/BiomedSciAI/biomed-multi-view) presented in their preprint [Multi-view biomedical foundation models for molecule-target and property prediction](https://arxiv.org/abs/2410.19704).

The model integrates multiple 'views' of molecules, namely, sequence (SMILES), image (RDkit), and graph (RDkit) representations, to learn a unified embedding that can be used for a variety of downstream tasks.

![MultiView diagram](docs/overview.png)

_Figure: Schematic of multi-view architecture. Embeddings from three single view pre-trained encoders (Image, Graph and Text) are combined by the aggregator module into a combined embedding. The network is finetunable for downstream predictive tasks._

# Our contribution

Here, we aim to extend and fine-tune the model for drug-target interaction (DTI) prediction between small molecules and proteins from the DAVIS and KIBA datasets.

Particularly, we aim to:
1. Unify the approach with findings from D. Illiadis' work on Multi-branch Neural Networks
     - [Multi‑target prediction for dummies using two‑branch neural networks](https://doi.org/10.1007/s10994-021-06104-5)
     - [A Comparison of Embedding Aggregation Strategies in Drug-Target Interaction Prediction](https://doi.org/10.1101/2023.09.25.559265)
2. Incorporate protein branche(s) to the model
     - Graph view inspired by the contact-map approach proposed by [M. Jiang et al.](https://doi.org/10.1039/d0ra02297g) and [R. Gorantla et al.](https://doi.org/10.1101/2023.09.25.559265)
     - DNA view
3. Incorporate a stochastic sampling strategy to the model through the use of variational autoencoders (VAEs) to enable:
      - Learning a structured latent space
      - Conditioned generation of novel drug molecules
4. Evaluate the model on the [DAVIS and KIBA datasets](https://tdcommons.ai/multi_pred_tasks/dti)


# Getting started with `biomed-multi-view-DTI-VAE`

## Installation

### Prerequisites
* Operating System: Linux or macOS
* Python Version: Python 3.11
* Conda: Anaconda or Miniconda installed
* Git: Version control to clone the repository

### Set-up commands
```bash
# Set up the root directory
export ROOT_DIR=~/biomed-multiview
mkdir -p $ROOT_DIR

# Create and activate the Conda environment
conda create -y python=3.11 --prefix $ROOT_DIR/envs/biomed-multiview
conda activate $ROOT_DIR/envs/biomed-multiview

# Clone the repository
mkdir -p $ROOT_DIR/code
cd $ROOT_DIR/code
git clone https://github.com/BiomedSciAI/biomed-multi-view.git
cd biomed-multi-view

# Install dependencies (non-macOS)
pip install -e .[dev]
pip install -r requirements.txt

# For macOS with Apple Silicon
noglob pip install -e .[dev]
pip install -r requirements-mps.txt

# Verify installation
python -m unittest bmfm_sm.tests.all_tests
```

## Get embeddings from the pretrained model

You can generate embeddings for a given molecule using the pretrained model with the following code. You can excute from cell 1 to 5 in the notebook run the same.

``` python
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel

# Example SMILES string for a molecule
example_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

# Obtain embeddings from the pretrained model
example_emb = SmallMoleculeMultiViewModel.get_embeddings(
    smiles=example_smiles,
    model_path="ibm/biomed.sm.mv-te-84m",
    huggingface=True,
)
print(example_emb)
```

This will output the embedding vector for the given molecule.

---
---
---

## Data preparation

To evaluate and train the models, you need to set up the necessary data, splits, configuration files, and model checkpoints. This section will guide you through the process.

#### Step 1: Set up the `$data_root` directory
First, create a directory to serve as your root directory for all the data. This directory will house all datasets, splits, configuration files, and model checkpoints.
```bash
mkdir -p $ROOT_DIR/data_root
```

#### Step 2: Set the environment variable
Set the `$BMFM_HOME` environment variable to point to your data root directory. This helps the scripts locate your data.
```bash
export BMFM_HOME=$ROOT_DIR/data_root
```
Optionally, add the export command to your shell configuration file (e.g., $HOME/.bashrc for bash):
```bash
echo 'export BMFM_HOME=$ROOT_DIR/data_root' >> $HOME/.bashrc
```

#### Step 3: Download the data splits, configuration files and checkpoints
We provide all the necessary data splits, configuration files, and model checkpoints in a single archive to simplify the setup process.

* Download `data_root_os_v1.tar.gz`: from [this location](https://ad-prod-biomed.s3.us-east-1.amazonaws.com/biomed.multi-view/data_root_os_v1.tar.gz).
* Extract the Archive: 	Uncompress the tar file into your data root directory
  ```bash
  tar -xzvf data_root_os_v1.tar.gz -C $BMFM_HOME
  ```
This will populate `$BMFM_HOME` with the required files and directories.

#### Step 4. Download MoleculeNet datasets
We provide a script to download the MoleculeNet datasets automatically that you can run:
```bash
run-download-moleculenet
```
This script will download the MoleculeNet datasets into `$BMFM_HOME/datasets/raw_data/MoleculeNet/`. **Note**: The `run-download-moleculenet` command launches a Python script that can be executed using `bmfm_sm.python -m bmfm_sm.launch.download_molecule_net_data` from `$ROOT_DIR/code/biomed-multi-view` directory as well.

#### Step 5. Verify the directory structure
After completing the above steps, your `$BMFM_HOME` directory should have the following structure:
```bash
$BMFM_HOME/
├── bmfm_model_dir
│   ├── finetuned
│   │   └── MULTIVIEW_MODEL
│   │       └── MoleculeNet
│   │           └── ligand_scaffold
│   └── pretrained
│       └── MULTIVIEW_MODEL
│           ├── biomed-smmv-base.pth
│           └── biomed-smmv-with-coeff-agg.pth
├── configs_finetune
│   └── MULTIVIEW_MODEL
│       └── MoleculeNet
│           └── ligand_scaffold
│               ├── BACE
│               ├── BBBP
│               ├── CLINTOX
│               ├── ESOL
│               ├── FREESOLV
│               ├── HIV
│               ├── LIPOPHILICITY
│               ├── MUV
│               ├── QM7
│               ├── SIDER
│               ├── TOX21
│               └── TOXCAST
└── datasets
    ├── raw_data
    │   └── MoleculeNet
    │       ├── bace.csv
    │       ├── bbbp.csv
    │       ├── clintox.csv
    │       ├── esol.csv
    │       ├── freesolv.csv
    │       ├── hiv.csv
    │       ├── lipophilicity.csv
    │       ├── muv.csv
    │       ├── qm7.csv
    │       ├── qm9.csv
    │       ├── sider.csv
    │       ├── tox21.csv
    │       └── toxcast.csv
    └── splits
        └── MoleculeNet
            └── ligand_scaffold
                ├── bace_split.json
                ├── bbbp_split.json
                ├── clintox_split.json
                ├── esol_split.json
                ├── freesolv_split.json
                ├── hiv_split.json
                ├── lipophilicity_split.json
                ├── muv_split.json
                ├── qm7_split.json
                ├── sider_split.json
                ├── tox21_split.json
                └── toxcast_split.json
```

After successfully completing the installation and data preparation steps, you are now ready to:

* Usage: Learn how to obtain embeddings from the pretrained model.
* Evaluation: Assess the model’s performance on benchmark datasets using our pretrained checkpoints.
* Training: Understand how to finetune the pretrained model using the provided configuration files.
* Inference: Run the model on sample data in inference mode to make predictions.


### Evaluation
We provide the `run-finetune` command for running finetuning and evaluation processes. You can use this command to evaluate the performance of the finetuned models on various datasets. **Note**: The `run-finetune` command launches a Python script that can be executed using `bmfm_sm.python -m launch.download_molecule_net_data` from `$ROOT_DIR/code/biomed-multi-view` directory as well.

To see the usage options for the script, run:
```bash
run-finetune --help
```

This will display:
``` bash
Usage: run-finetune [OPTIONS]

Options:
  --model TEXT           Model name
  --dataset-group TEXT   Dataset group name
  --split-strategy TEXT  Split strategy name
  --dataset TEXT         Dataset name
  --fit                  Run training (fit)
  --test                 Run testing
  -o, --override TEXT    Override parameters in key=value format (e.g.,
                         trainer.max_epochs=10)
  --help                 Show this message and exit.
```

To evaluate a finetuned checkpoint on a specific dataset, use the --test option along with the --dataset parameter. For example, to evaluate on the BBBP dataset:

``` bash
python run-finetune --test --dataset BBBP
```
If you omit the `--dataset` option, the script will prompt you to choose a dataset:
```bash
python run-finetune --test
Please choose a dataset (FREESOLV, BBBP, CLINTOX, MUV, TOXCAST, QM9, BACE, LIPOPHILICITY, ESOL, HIV, TOX21, SIDER, QM7): BBBP
```
This command will evaluate the finetuned checkpoint corresponding to the BBBP dataset using the test set of the `ligand_scaffold` split.


### Training (finetuning)
To finetune the pretrained model on a specific dataset, use the --fit option:

``` bash
python run-finetune --fit --dataset BBBP
```
Again, if you omit the --dataset option, the script will prompt you to select one:

```bash
python run-finetune --fit
Please choose a dataset (FREESOLV, BBBP, CLINTOX, MUV, TOXCAST, QM9, BACE, LIPOPHILICITY, ESOL, HIV, TOX21, SIDER, QM7): BBBP
```
This command will start the finetuning process for the BBBP dataset using the configuration files provided in the `configs_finetune` directory.

Note: You can override default parameters using the `-o` or `--override` option. For example:

```bash
python run-finetune --fit --dataset BBBP -o trainer.max_epochs=10
```

**Note**: If you run into out of memory errors, you can reduce the batch size using the following syntax

```bash
python run-finetune --fit --dataset BBBP -o data.init_args.batch_size=4
```

# Citations

```
@misc{suryanarayanan2024multiviewbiomedicalfoundationmodels,
      title={Multi-view biomedical foundation models for molecule-target and property prediction},
      author={Parthasarathy Suryanarayanan and Yunguang Qiu and Shreyans Sethi and Diwakar Mahajan and Hongyang Li and Yuxin Yang and Elif Eyigoz and Aldo Guzman Saenz and Daniel E. Platt and Timothy H. Rumbell and Kenney Ng and Sanjoy Dey and Myson Burch and Bum Chul Kwon and Pablo Meyer and Feixiong Cheng and Jianying Hu and Joseph A. Morrone},
      year={2024},
      eprint={2410.19704},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2410.19704},
}
```
