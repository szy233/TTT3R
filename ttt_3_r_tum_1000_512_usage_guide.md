# TTT3R Evaluation Guide (TUM 1000×512 Sensitivity)

This guide explains how to reproduce the sensitivity experiments for TTT3R using the TUM RGB-D dataset with the formal evaluation setting:

- Sequence length: 1000 frames
- Resolution: 512
- Model: `ttt3r_joint`
- Grid: `tau ∈ {0.5, 1, 2, 4}`, `freq_cutoff ∈ {2, 4, 8}`

## 1. Environment Setup

Tested environment:

- Python 3.10
- CUDA 11+
- PyTorch 2.0+

Create conda environment:

```bash
conda create -n ttt3r python=3.10
conda activate ttt3r
```

Install dependencies:

```bash
pip install torch torchvision
pip install accelerate
pip install matplotlib numpy tqdm
pip install huggingface_hub
pip install -r requirements.txt
```

Set Python path:

```bash
cd ~/TTT3R
export PYTHONPATH=$(pwd)/src
```

## 2. Clone the Repository

```bash
git clone <your_repo_url>
cd TTT3R
```

## 3. Download Model Weights

Download the pretrained checkpoint:

```text
cut3r_512_dpt_4_64.pth
```

Place it at:

```text
TTT3R/model/cut3r_512_dpt_4_64.pth
```

Expected structure:

```text
TTT3R/
├── model/
│   └── cut3r_512_dpt_4_64.pth
```

## 4. Download the TUM RGB-D Dataset

Download this sequence:

```text
rgbd_dataset_freiburg3_long_office_household
```

Official download page:

```text
https://vision.in.tum.de/data/datasets/rgbd-dataset/download
```

Example commands:

```bash
mkdir -p ~/TTT3R/data/tum
cd ~/TTT3R/data/tum
wget -c https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
tar -xvzf rgbd_dataset_freiburg3_long_office_household.tgz
```

Expected structure after extraction:

```text
TTT3R/data/tum/rgbd_dataset_freiburg3_long_office_household/
├── rgb/
├── rgb.txt
├── groundtruth.txt
```

## 5. Preprocess the Dataset

Make sure `datasets_preprocess/long_prepare_tum.py` uses the local path:

```python
dirs = glob.glob("./data/tum/*/")
```

Then run preprocessing:

```bash
cd ~/TTT3R
python datasets_preprocess/long_prepare_tum.py
```

This will generate:

```text
TTT3R/data/long_tum_s1/
```

Expected structure:

```text
TTT3R/data/long_tum_s1/
└── rgbd_dataset_freiburg3_long_office_household/
    ├── rgb_50/
    ├── rgb_100/
    ├── ...
    ├── rgb_1000/
    ├── groundtruth_50.txt
    ├── groundtruth_100.txt
    ├── ...
    └── groundtruth_1000.txt
```

## 6. Formal Sensitivity Sweep Setting

The formal sweep uses:

- Dataset: `tum_s1_1000`
- Resolution: `512`
- Update type: `ttt3r_joint`
- Spectral temperature: `1.0`
- Tau grid: `{0.5, 1.0, 2.0, 4.0}`
- Freq cutoff grid: `{2, 4, 8}`

Total runs:

```text
4 × 3 = 12 experiments
```

## 7. Run the Sensitivity Experiments

Run the formal script:

```bash
cd ~/TTT3R
chmod +x eval/run_sensitivity_tum1000_512.sh
nohup bash eval/run_sensitivity_tum1000_512.sh > eval/sensitivity_tum1000_512.log 2>&1 &
```

Check progress:

```bash
tail -f eval/sensitivity_tum1000_512.log
```

Results will be stored in:

```text
eval_results/relpose/sensitivity/tum/
```

Example output directories:

```text
eval_results/relpose/sensitivity/tum/
├── ttt3r_joint_tau0.5_c2/
├── ttt3r_joint_tau0.5_c4/
├── ttt3r_joint_tau0.5_c8/
├── ttt3r_joint_tau1.0_c2/
├── ...
```

Each experiment directory should contain files such as:

```text
rgbd_dataset_freiburg3_long_office_household_eval_metric.txt
rgbd_dataset_freiburg3_long_office_household_traj_error.png
rgbd_dataset_freiburg3_long_office_household/pred_traj.txt
```

## 8. Run the Analysis Script

After all 12 runs are finished:

```bash
cd ~/TTT3R
python analysis/sensitivity_analysis_tum1000.py
```

Outputs will be saved to:

```text
analysis_results/sensitivity_tum1000/
```

Expected files:

```text
sensitivity_table.txt
sensitivity_table.tex
sensitivity_heatmap.pdf
sensitivity_curves.pdf
```

## 9. Quick Sanity Checks

Check whether all 12 runs completed:

```bash
find eval_results/relpose/sensitivity/tum -name "*_eval_metric.txt"
```

Check generated analysis files:

```bash
ls analysis_results/sensitivity_tum1000
```

View the final text table:

```bash
cat analysis_results/sensitivity_tum1000/sensitivity_table.txt
```

## 10. Notes

- This setup is intended for the formal TUM sensitivity experiment at `1000 frames + 512 resolution`.
- The repository must contain the script:

```text
eval/run_sensitivity_tum1000_512.sh
```

- The repository must also contain the analysis script:

```text
analysis/sensitivity_analysis_tum1000.py
```

- If GPU memory is insufficient on a local machine, run this formal setting on a server GPU with larger memory.

## 11. Minimal Reproduction Order

```bash
conda activate ttt3r
cd ~/TTT3R
export PYTHONPATH=$(pwd)/src
python datasets_preprocess/long_prepare_tum.py
nohup bash eval/run_sensitivity_tum1000_512.sh > eval/sensitivity_tum1000_512.log 2>&1 &
python analysis/sensitivity_analysis_tum1000.py
```

