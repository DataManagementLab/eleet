# Project Structure

- Code regarding Pre-training (i.e. corpus construction and pre-training scripts) is located in "eleet_pretrain"
- Code for everything else (e.g. query plans, MMOps, baselines, benchmark) is located in "eleet".
- Some scripts are located in "scripts" and "slurm" as described below.

# How to install

1. Use Python version 3.8
2. Install PyTorch https://pytorch.org/get-started/locally/  (tested using conda)
3. Install torch-scatter: https://github.com/rusty1s/pytorch_scatter  (tested using conda)

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

4. Install Cython: ```pip install Cython```
5. Install PyJinius: ```conda install -c conda-forge pyjnius```
6. Install FastBPE: ```conda install -c conda-forge fastbpe```
7. Install curl: ```conda install curl```
8. Install other stuff: ```pip install -r requirements.txt```
9. Install: ```pip install -e .```
10. Install TaBERT: ```cd TaBERT/ && pip install -e . && cd ..```
11. Download English Language for spacy: ```python -m spacy download en_core_web_sm```

# Pre-training

12. Run MongoDB and set environment variables (MONGO_USER, MONGO_PASSWORD, MONGO_HOST, MONGO_PORT, MONGO_DB)
    https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
13. Start data pre-processing: python scripts/load_data.py trex-wikidata
    --> preprocessed data will appear in datasets/preprocessed_data/preprocessed_trex-wikidata*
14. Use slurm/pretrain.slurm for pre-training (Adjust path in file first).
    --> Will store pretrained model in models/pretrained

# Finetuning + Evaluation

15. Generate TREx Dataset: ```python eleet/datasets/trex/generate.py```
16. Generate Rotowire Dataset: ```python eleet/datasets/rotowire/generate.py```
17. Run finetuning: ```sbatch slurm/rotowire/train-ours.slurm``` (Repeat for other datasets and models).
    --> Will store finetuned model in models/rotowire/ours/finetuned
18. Run evaluation: ```python eleet/benchmark.py --slurm-mode --use-test-set```
19. Visualize results using Jupyter notebooks located in ```scripts/*.ipynb```
