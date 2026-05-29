

https://github.com/user-attachments/assets/f4a4cda7-9fd9-4586-a0f4-b02ec5ae80d3

# AdapSuckNet: An Adaptive Suction Pose Estimation Network for Robotic Grasping

This repository contains the official PyTorch implementation of the paper **"AdapSuckNet: An Adaptive Suction Pose Estimation Network for Robotic Grasping"**.

AdapSuckNet is an end-to-end heterogeneous dual-stream deep network designed for robust and precise 3D suction pose estimation of retail commodities in complex, highly cluttered scenarios.

---

## 1. Overview
Object pose estimation offers considerable potential for enhancing robot sorting efficiency in automated logistics and retail scenarios. While current deep learning-based suction pose estimation methods show significant progress on synthetic datasets and standard benchmarks, they remain constrained by rudimentary multi-modal fusion, insufficient semantic guidance, and physical geometric feasibility.

To address these limitations, **AdapSuckNet** introduces:
- **Heterogeneous Dual-Stream Architecture**: Seamlessly processes 2D grayscale images for instance segmentation (via YOLOv11) and extracts regional 3D point clouds for localized geometric and category feature reasoning.
- **Learnable Dynamic Gating Module**: Adaptively adjusts fusion weights (W_gate^1 in R^{128 x 512}, W_gate^2 in R^{2 x 128}) via a multi-layer perceptron (FC: 512 -> 128 -> 2) to dynamically integrate semantic and geometric characteristics based on situational context.
- **Multi-Head Self-Attention (MHSA)**: Captures global spatial and topological dependencies across the joint feature space.
- **Multi-Task Prediction Head & Geometric Regularization**: Jointly optimizes 3D coordinates (P in R^{B x K x 3}), surface normal vectors (N in R^{B x K x 3}), and grasp confidence scores (C in R^{B x K}). The training is constrained by a physical geometric loss (L_Geo = L_dist + \eta L_stab) to prevent mechanical interference and align the suction direction against gravity.

---

## 2. Key Mathematical Formulations

### 2.1 Dynamic Gating Feature Fusion
The gating weight vector g is dynamically generated using semantic features f_sem and geometric features f_geo:
g = softmax(W_gate^2 * ReLU(W_gate^1 * [f_geo; f_sem] + b_gate^1) + b_gate^2) in R^{B x 2}

### 2.2 Physical Geometric Constraint Loss
To ensure physical feasibility and suction stability, the geometric loss enforces distance clearances and normal alignment against the gravity vector z = (0,0,1)^T:
L_dist = (1 / (K(K-1))) * sum_{i=1}^{K} sum_{j!=i}^{K} max(0, d_min - ||p_i - p_j||_2)
L_stab = (1 / K) * sum_{i=1}^{K} max(0, cos(\theta_max) - n_i^T z)
L_Geo = L_dist + \eta L_stab

### 2.3 Total Objective Function
The network is optimized end-to-end via the total multi-task loss using Hungarian bipartite matching \hat{M}:
L_total = L_Hung + \lambda L_Geo
L_conf = -(1 / K) * sum_{k=1}^{K} [y_k log(c_k) + (1-y_k) log(1-c_k)]
L_pos = (1 / |M|) * sum_{(k,m) in M} (||p_k - g_m||_2 + (1 - n_k^T * n_m))

---

## 3. Dataset & Demonstration Videos

### 3.1 Dataset Preparation
The training and evaluation pipeline utilizes a hybrid dataset strategy:
1. **Self-Collected Dataset**: Consists of more than 10 types of typical retail commodities, with approximately 300 groups of raw multi-modal samples collected for each category.
2. **SuctionNet-1Billion**: Incorporated as an external benchmark supplement to expand structural and geometric diversity. Please download the public data from the [SuctionNet Official Baseline Repository](https://github.com/graspnet/suctionnet-baseline).

> 📥 **[Download Our Processed Dataset & Annotations (Google Drive Folder)](https://drive.google.com/drive/folders/1xgQDDPR9inqq-rhK0mxjQe_yFEPt5ejH?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)**

### 3.2 Demonstration Videos
We provide complete multi-scenario experimental videos demonstrating AdapSuckNet deployed on a real UR5e robot hardware setup for sorting high-density stacked items.
> 🎬 **[Watch the Hardware Deployment Demo Video (Google Drive Folder)](https://drive.google.com/drive/folders/1xgQDDPR9inqq-rhK0mxjQe_yFEPt5ejH?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)**

---

## 4. Main Experimental Results

### 4.1 Ablation Analysis (Section 4.1)
Evaluated on our test benchmark to check individual component contributions under Vacuum Suction Index (VSI >= 0.75):

| Network | AP | AP_0.8 | AP_0.5 |
| :--- | :---: | :---: | :---: |
| w/o Self-Attention | 65.58 (-14.91%) | 59.27 (-15.63%) | 69.14 (-17.37%) |
| w/o Gated Fusion | 69.70 (-9.56%) | 64.55 (-8.11%) | 75.11 (-10.24%) |
| w/o Geometric Constraint | 72.23 (-6.28%) | 66.26 (-5.68%) | 77.19 (-7.76%) |
| **Complete Model (Ours)** | **77.07** | **70.25** | **83.68** |

### 4.2 Benchmark Comparisons (Section 4.2)
Comparative precision metrics across Seen, Similar, and Novel object categories:

| Network Model | Seen (AP / AP_0.8 / AP_0.5) | Similar (AP / AP_0.8 / AP_0.5) | Novel (AP / AP_0.8 / AP_0.5) |
| :--- | :---: | :---: | :---: |
| DexNet 3.0 | 85.8 / 78.9 / 89.8 | 71.2 / 66.5 / 77.1 | 59.1 / 52.7 / 67.3 |
| SuctionNet | 87.1 / 79.5 / 91.0 | 73.3 / 67.5 / 81.4 | 62.6 / 54.1 / 71.9 |
| **Ours (AdapSuckNet)** | **89.5 / 82.3 / 92.7** | **75.9 / 67.9 / 83.5** | **64.4 / 57.4 / 74.6** |

### 4.3 Real-World Cluttered Bin Grasping Tests (Section 4.3)
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

## 5. Data and Code Availability Statement
The datasets used during the current study include a publicly available benchmark and a self-collected dataset. The public benchmark can be accessed via the SuctionNet official baseline repository ([https://github.com/graspnet/suctionnet-baseline.git]), which provides standardized access to large-scale suction grasping data. The original sources of this dataset are maintained by its respective providers. The complete model code, our processed dataset, and demonstration videos that support the experimental findings of this study are openly available in a public repository at: [https://github.com/IRM-Lab/AdapSuckNet]. Further information is available from the corresponding author on reasonable request.
