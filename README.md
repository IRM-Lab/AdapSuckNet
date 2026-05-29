---

## 4. Dataset & Demonstration Videos

### 4.1 Dataset Preparation
The training and evaluation pipeline utilizes a hybrid dataset strategy:
1. **Self-Collected Dataset**: Consists of more than 10 types of typical retail commodities, with approximately 300 groups of raw multi-modal samples collected for each category.
2. **SuctionNet-1Billion**: Incorporated as an external benchmark supplement to expand structural and geometric diversity. Please download the public data from the [SuctionNet Official Baseline Repository](https://github.com/graspnet/suctionnet-baseline).

> 📥 **[Download Our Processed Dataset & Annotations (Google Drive Folder)](https://drive.google.com/drive/folders/1xgQDDPR9inqq-rhK0mxjQe_yFEPt5ejH?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)**

### 4.2 Demonstration Videos
We provide complete multi-scenario experimental videos demonstrating AdapSuckNet deployed on a real UR5e robot hardware setup for sorting high-density stacked items.
> 🎬 **[Watch the Hardware Deployment Demo Video (Google Drive Folder)](https://drive.google.com/drive/folders/1xgQDDPR9inqq-rhK0mxjQe_yFEPt5ejH?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)**

---

## 5. Main Experimental Results

### 5.1 Ablation Analysis (Section 4.1)
Evaluated on our test benchmark to check individual component contributions under Vacuum Suction Index (VSI >= 0.75):

| Network | AP | AP_0.8 | AP_0.5 |
| :--- | :---: | :---: | :---: |
| w/o Self-Attention | 65.58 (-14.91%) | 59.27 (-15.63%) | 69.14 (-17.37%) |
| w/o Gated Fusion | 69.70 (-9.56%) | 64.55 (-8.11%) | 75.11 (-10.24%) |
| w/o Geometric Constraint | 72.23 (-6.28%) | 66.26 (-5.68%) | 77.19 (-7.76%) |
| **Complete Model (Ours)** | **77.07** | **70.25** | **83.68** |

### 5.2 Benchmark Comparisons (Section 4.2)
Comparative precision metrics across Seen, Similar, and Novel object categories:

| Network Model | Seen (AP / AP_0.8 / AP_0.5) | Similar (AP / AP_0.8 / AP_0.5) | Novel (AP / AP_0.8 / AP_0.5) |
| :--- | :---: | :---: | :---: |
| DexNet 3.0 | 85.8 / 78.9 / 89.8 | 71.2 / 66.5 / 77.1 | 59.1 / 52.7 / 67.3 |
| SuctionNet | 87.1 / 79.5 / 91.0 | 73.3 / 67.5 / 81.4 | 62.6 / 54.1 / 71.9 |
| **Ours (AdapSuckNet)** | **89.5 / 82.3 / 92.7** | **75.9 / 67.9 / 83.5** | **64.4 / 57.4 / 74.6** |

### 5.3 Real-World Cluttered Bin Grasping Tests (Section 4.3)
Summary of physical validation on hardware over 12 experimental groups:

| Num Objects | Num Groups | Num Attempts | Num Successes | % Success | % Removed |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6 | 2 | 14 | 12 | 85.71% | 100.00% |
| 10 | 2 | 22 | 20 | 90.91% | 100.00% |
| 12 | 2 | 27 | 24 | 88.89% | 100.00% |
| 15 | 2 | 35 | 28 | 80.00% | 93.33% |
| 18 | 2 | 41 | 34 | 82.93% | 94.44% |
| 20 | 2 | 42 | 37 | 88.10% | 92.50% |
| **Total** | **12** | **181** | **155** | **85.64%** | **95.68%** |

---

## 6. Data and Code Availability Statement
The datasets used during the current study include a publicly available benchmark and a self-collected dataset. The public benchmark can be accessed via the SuctionNet official baseline repository ([https://github.com/graspnet/suctionnet-baseline.git]), which provides standardized access to large-scale suction grasping data. The original sources of this dataset are maintained by its respective providers. The complete model code, our processed dataset, and demonstration videos that support the experimental findings of this study are openly available in a public repository at: [https://github.com/IRM-Lab/AdapSuckNet]. Further information is available from the corresponding author on reasonable request.
"""
with open("README_clean.txt", "w", encoding="utf-8") as f:
    f.write(content)
print("Clean txt file generated.")
