# ELEET: Efficient Learned Query Execution over Text and Tables

ELEET is an Execution Engine to run multi-modal queries over datasets containing texts and tables.
This is the implementation described in 

> Matthias Urban and Carsten Binnig: "ELEET: Efficient Learned Query Execution over Text and Tables.", PVLDB, 17(13): 4867-4880, 2024. [[PDF]]()
>
> ![Image of ELEET](paper_img.png)



# Project Structure

- Code regarding Pre-training (i.e. corpus construction and pre-training scripts) is located in "eleet_pretrain"
- Code for everything else (e.g. query plans, MMOps, baselines, benchmark) is located in "eleet".
- Some scripts are located in "scripts" and "slurm" as described below.

# How to install

1. `git clone git@github.com:DataManagementLab/eleet.git`
1. `cd eleet`
1. `git submodule update --init`
1. Use Python version 3.8 (e.g., by using conda): `conda create -n eleet` then `conda install python=3.8 pip`
1. Install PyTorch https://pytorch.org/get-started/locally/
1. Install torch-scatter: https://github.com/rusty1s/pytorch_scatter

   Versions we used:

   ```
   $ conda list | grep torch
   pytorch                   1.12.1          py3.8_cuda11.3_cudnn8.3.2_0    pytorch
   pytorch-mutex             1.0                        cuda    pytorch
   pytorch-scatter           2.0.9           py38_torch_1.12.0_cu113    pyg
   torch                     1.12.0+cpu               pypi_0    pypi
   torch-scatter             2.0.7                    pypi_0    pypi
   torchaudio                0.12.0+cpu               pypi_0    pypi
   torchvision               0.13.0+cpu               pypi_0    pypi
   ```

1. Some requirements need to be installed using conda, as pip seems to have problems:
    1. Install Cython: ```conda install Cython```
    1. Install Spacy: ```conda install spacy```
    1. Install PyJinius: ```conda install -c conda-forge pyjnius```
    1. Install FastBPE: ```conda install -c conda-forge fastbpe```
    1. Install curl: ```conda install curl```
1. Install other requirements: ```pip install -r requirements.txt```
1. Install: ```pip install -e .```
1. Install TaBERT: ```cd TaBERT/ && pip install -e . && cd ..```
1. Download English Language for spacy: ```python -m spacy download en_core_web_sm```

# Pre-training

1. Run MongoDB and set environment variables (MONGO_USER, MONGO_PASSWORD, MONGO_HOST, MONGO_PORT, MONGO_DB)
    https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
1. Start data pre-processing: python scripts/load_data.py trex-wikidata
    --> preprocessed data will appear in datasets/preprocessed_data/preprocessed_trex-wikidata*
1. Use slurm/pretrain.slurm for pre-training (Adjust path in file first).
    --> Will store pretrained model in models/pretrained

# Finetuning + Evaluation

15. Generate TREx Dataset: ```python eleet/datasets/trex/generate.py```
16. Generate Rotowire Dataset: ```python eleet/datasets/rotowire/generate.py```
17. Run finetuning: ```sbatch slurm/rotowire/train-ours.slurm``` (Repeat for other datasets and models).
    --> Will store finetuned model in models/rotowire/ours/finetuned
18. Run evaluation: ```python eleet/benchmark.py --slurm-mode --use-test-set```
19. Visualize results using Jupyter notebooks located in ```scripts/*.ipynb```


# Reference

If you use code or the benchmarks of this repository then please cite our paper:

```bib
@inproceedings{eleet,
  title={ELEET: Efficient Learned query Execution over Text and Tables},
  author = {Matthias Urban and Carsten Binnig},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={13},
  pages={4867--4880},
  year={2024},
  publisher={VLDB Endowment}
}

```
