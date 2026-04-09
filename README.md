# CONSERVAttack
GitHub repository of the accompanying [Shapes are not enough: CONSERVAttack and its use for finding vulnerabilities and uncertainties in machine learning applications](https://arxiv.org/abs/2603.13970) paper. 

CONSERVAttack is a constraint-preserving adversarial attack framework that generates adversarial samples while minimising marginal distribution and linear correlation shifts.

---

## Data

In the original paper, three different datasets were discussed:

1. The DonutDummy dataset can be produced in this pipeline (`Data/MakeDonutDummyData.py`).
2. The Higgs dataset was taken from the [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/competitions/higgs-boson) Kaggle challenge.
3. The TTvsWW dataset can be found on [HuggingFace](https://huggingface.co/datasets/TSaala/TTvsWW/tree/main).

---

## Requirements

**Tested with:**
- Python 3.12
- CUDA 12.6 (toolkit) + compatible NVIDIA driver

**Python packages** (install via `pip install -r requirements.txt`):

```
numpy
matplotlib
pandas
pyarrow
scipy
scikit-learn
cupy-cuda12x
tensorflow
tqdm
keras
torch
```
---

## Detector Pipeline

A runscript for the full detector pipeline can be found in `run_detector_pipeline.py`. There, a few parameters should be adjusted:

- N_RUNS (int): How many independent detector runs should be done for the cross run test
- RUN_PREFIX (str): Name of the experiment (e.g. DonutDetector)
- NUM_SIGNAL (int): How many signal events should be generated for the DonutDummy dataset
- NUM_BACKGROUND: How many background events should be generated for the DonutDummy dataset
- TRAIN_ADV_SIZE: How many training adversaries should be generated for the detector pipeline
- VAL_ADV_SIZE: How many validation adversaries should be generated for the detector pipeline
- TEST_ADV_SIZE: How many test adversaries should be generated for the detector pipeline


> **Note:** For further adjustments - such as adjusting the attack hyperparameters, or the hyperparameters of the base line or the detector network - the respective code blocks in `Detector/DetectorPipeline.py` need to be adjusted.

---

## Attack Pipeline

### Step 1 — Generate dummy data

```bash
python Data/MakeDonutDummyData.py
```

This generates a 2D toy dataset saved to `Data/donut_signal_background.csv`:

- **Signal** (label `0`): 50 000 samples drawn from a 2D Gaussian centered at the origin (σ = 0.5)
- **Background** (label `1`): 50 000 samples drawn from a ring distribution at radius r = 1.8 (σ = 0.5)

> **Note:** The script saves to `donut_signal_background_smallRadius.csv` by default. Rename the output to `donut_signal_background.csv` before running Step 2, or update the filename in `RunAttack.py`.
Additionally, it is possible to adjust the parameters of the dummy distributions, as well as the amount of events to be generated. 

---

### Step 2 — Run the attack

```bash
cd Attack
python RunAttack.py
```

`RunAttack.py` does the following:

1. Loads `Data/donut_signal_background.csv`
2. Trains a small MLP classifier (Dense 32 → 16 → 1) on the data, or loads an existing model from `Models/best_model.keras`
3. Selects correctly classified background test samples
4. Calls `generate_adversarial_samples` from `Helpers/MultiGPUCONSERVAttack.py` to generate adversarial examples
5. Saves adversarial samples to `Results/adversarial_samples_bg.feather`
6. Produces diagnostic plots in `Results/`

Outputs are written to:

```
Results/
Models/
```

> **Note:** Within the `RunAttack.py` script, all attack parameters are fully adjustable. Further, it is also possible to adjust or replace the deep learning model used for the classification.

Additionally, this approach serves as a template to extrapolate to other datasets, simply adjust the data loading, as well as - if desired - the classification model. Then tune the attack parameters as necessary.

---

## Project structure

```
CONSERVAttack/
├── Data/
│   ├── MakeDonutDummyData.py        
├── Helpers/
│   ├── __init__.py
│   └── MultiGPUCONSERVAttack.py     
├── Attack/
│   └── RunAttack.py                 
├── Models/      
├── Results/
└── requirements.txt
```
