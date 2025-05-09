# HPML Project: HiSS with Mamba2
## Team Information
- **Team Name**: wisebees
- **Members**:
- Albert Wen (aw3575)
- Nicholas Yah (nzy2000)
- Xian Jiang (xj2281)
---
## 1. Problem Statement
Hierarchical State Space Models (HiSS) is an architecture that leverages models like S4, Mamba for continuous sequence prediction. The paper can be found here: https://hiss-csp.github.io/. After reading the paper, we found that the authors did not analyze the training runtime of the models. Additionally, Mamba2 was developed after the paper, which motivated us to investigate the questions below.

How does the inclusion of Mamba2 affect the performance of HiSS?
How does Mamba2 perform on different sensory datasets?
Can we attribute any performance gain or loss to optimizations from Mamba to Mamba2?

We utilize the existing HiSS architecture but make modifications to include the Mamba2 model. Our solution provides empirical evidence on whether Mamba2 optimizations improves performance when applied in the context of sensory data and hierarchical models.

---
## 2. Model Description
Summarize the model architecture(s) used (e.g., ResNet-18, Transformer).
- Framework: PyTorch
- Hierarchical Model Structure
- Models involved: Mamba-1, Mamba-2, Transformer, LSTM, S4
---
## 3. Final Results Summary
MSE-loss and Runtime table:

| High-level | Low-level | Val MSE | Training Runtime |
|----------------------|-------------|-------------|-------------|
| Mamba2 | Transformer | NIL: hitting errors | NIL: hitting errors |
| Mamba2 | LSTM | 0.0269 | 60.35 |
| Mamba2 | S4 | NIL: gives NAN | NIL: gives NAN |
| Mamba2 | Mamba | 0.0274 | 270.39 |
| Transformer | Mamba2 | 0.0299 | 1020.36 |
| LSTM | Mamba2 | 0.0269 | 1890.00 |
| S4 | Mamba2 | 0.0285 | 1215.37 |
| Mamba | Mamba2 | 0.0273 | 1500.33 |
| Mamba2 | Mamba2 | 0.0269 | 240.34 |



Models achieving best MSE loss: Mamba2 - LSTM, LSTM - Mamba2, Flat Mamba2

---
## 4. Reproducibility Instructions
### A. Requirements
Install dependencies:
```bash
conda env create -f env.yml
conda activate vt_state
```
If this causes dependency issues, do the installations manually and in order.
Build Mamba from source: https://github.com/state-spaces/mamba by cloning repo then
```bash
pip install .
```
Build causal-conv1d from source: https://github.com/Dao-AILab/causal-conv1d by cloning repo then
```bash
pip install .
```
---
### B. Wandb Dashboard
View training and evaluation metrics here: https://wandb.ai/aw3575-columbia-university?shareProfileType=copy

---
### C. Specify for Training or For Inference or if Both
To train the models, e.g.:
```bash
python train.py --config-name [name of config (e.g. vector_2_lstm)]
```
| High-level | Low-level | Corresponding config file | 
|----------------------|-------------|-------------|
| Mamba2 | Transformer | hl_mamba2_ll_transformer.yaml |
| Mamba2 | LSTM |  hl_mamba2_ll_lstm.yaml |
| Mamba2 | S4 |  hl_mamba2_ll_s4.yaml |
| Mamba2 | Mamba | hl_mamba2_ll_mamba.yaml |
| Transformer | Mamba2 | hl_transformer_ll_mamba2.yaml |
| LSTM | Mamba2 | hl_lstm_ll_mamba2.yaml |
| S4 | Mamba2 | hl_s4_ll_mamba2.yaml |
| Mamba | Mamba2 | hl_mamba_ll_mamba2.yaml |
| Mamba2 | Mamba2 | hl_mamba2_ll_mamba2.y|

| Flat Model | Corresponding config file | 
| Mamba2 | flat_mamba.yaml |


---
### D. Evaluation
To evaluate the trained model:
```bash
python train.py --config-name vector_lstm_2
```
---
### E. Quickstart: Minimum Reproducible Result
To reproduce our minimum reported result (e.g., XX.XX% accuracy), run:
```bash
# Step 1: Set up environment
See 4A for instructions on setting up environment
# Step 2: Download dataset
Refer to `data_processing/README.md` on detailed instructions
# Step 3: Prepare dataset
Refer to `data_processing/README.md` on detailed instructions
Run:
``` bash
python data_processing/process_vector_data.py
python create_dataset.py --config-name vector_config
```
# Step 4: Run training

```
python train.py --config-name vector_lstm_2
```
---
## 5. Notes (up to you)
- All config files are located in `conf/` directory
freq_ratio: 5
    freq_ratio: 5

