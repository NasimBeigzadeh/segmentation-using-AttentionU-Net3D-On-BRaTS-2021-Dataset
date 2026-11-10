# 3D Attention U-Net for Brain Tumor Segmentation on BraTS 2021

## Overview

This project focuses on **3D brain tumor segmentation using an Attention U-Net architecture** on the BraTS 2021 dataset.

The model performs volumetric segmentation of brain tumor subregions from MRI scans. The project includes model training, full-volume inference, and quantitative and qualitative evaluation on the test dataset.

The evaluation covers three clinically relevant tumor regions:

- **Whole Tumor (WT)**
- **Tumor Core (TC)**
- **Enhancing Tumor (ET)**

## Key Features

- 3D Attention U-Net for volumetric brain tumor segmentation.
- Full-volume inference on 3D MRI scans.
- Multi-region evaluation (WT, TC, and ET).
- Quantitative evaluation using Dice, Precision, Recall, IoU, and HD95.
- Qualitative visualization of segmentation predictions across three anatomical views.
- Patient-level test results summarized across the test cohort.

## Qualitative Results

The following figure presents segmentation examples from three test samples.

For each sample, the visualization includes three anatomical views and four comparison columns:

1. Original FLAIR MRI.
2. Ground-truth segmentation.
3. Model prediction.
4. Model prediction with ground-truth boundaries.

Tumor subregions are visualized using the following color scheme:

- **Red:** NCR/NET
- **Green:** Edema (ED)
- **Yellow:** Enhancing Tumor (ET)
- **White dashed contours:** Ground-truth boundaries

![Qualitative brain tumor segmentation results](results/figures/output.png)

*Figure 1. Qualitative full-volume segmentation results for three BraTS 2021 test samples.*

## Quantitative Results

The model was evaluated using full-volume inference on **188 test patients**.

The following table reports the mean and standard deviation of each evaluation metric across the test patients.

| Region | Metric | Mean ± Std | Valid Patients |
|---|---|---:|---:|
| WT | Dice | 0.9174 ± 0.0877 | 188/188 |
| WT | Precision | 0.9094 ± 0.1011 | 188/188 |
| WT | Recall | 0.9369 ± 0.0948 | 188/188 |
| WT | IoU | 0.8570 ± 0.1215 | 188/188 |
| WT | HD95 | 7.2447 ± 12.8347 | 188/188 |
| TC | Dice | 0.8937 ± 0.1521 | 188/188 |
| TC | Precision | 0.8971 ± 0.1632 | 188/188 |
| TC | Recall | 0.9174 ± 0.1357 | 188/188 |
| TC | IoU | 0.8332 ± 0.1880 | 188/188 |
| TC | HD95 | 5.0911 ± 11.8568 | 188/188 |
| ET | Dice | 0.8584 ± 0.1774 | 187/188 |
| ET | Precision | 0.8706 ± 0.1642 | 187/188 |
| ET | Recall | 0.8732 ± 0.1886 | 187/188 |
| ET | IoU | 0.7819 ± 0.1976 | 187/188 |
| ET | HD95 | 3.6212 ± 9.9397 | 185/188 |

**Notes:**
- Results are reported as mean ± standard deviation across valid test cases.
- The number of valid patients varies by metric because some cases may not have valid values for a particular metric.
- HD95 values are reported in voxel units unless physical voxel spacing is incorporated into the distance calculation.
- These results are based on full-volume test evaluation, rather than patch-level validation.

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Dice Similarity Coefficient (DSC):** Measures the overlap between predicted and ground-truth segmentation masks.
- **Precision:** Measures the proportion of predicted tumor voxels that are correct.
- **Recall:** Measures the proportion of ground-truth tumor voxels detected by the model.
- **Intersection over Union (IoU):** Measures the intersection relative to the union of predicted and ground-truth regions.
- **95th Percentile Hausdorff Distance (HD95):** Measures the 95th percentile of surface distances between predicted and ground-truth boundaries.

## Tumor Region Definitions

The three evaluation regions are defined as follows:

- **Whole Tumor (WT):** Includes all tumor subregions.
- **Tumor Core (TC):** Includes NCR/NET and ET.
- **Enhancing Tumor (ET):** Includes the enhancing tumor region.

The segmentation labels used in the evaluation are assumed to follow the preprocessing label mapping in which the original BraTS enhancing-tumor label is represented as label 3.

## Training Configuration

The training configuration used in the provided training code includes:

| Parameter | Value |
|---|---:|
| Model | 3D Attention U-Net |
| Dataset | BraTS 2021 |
| Batch Size | 3 |
| Learning Rate | 0.0001 |
| Weight Decay | 0.00001 |
| Target Epochs | 20 |
| Random Seed | 2026 |
| Mixed Precision | Enabled on CUDA |

## Repository Contents

The repository contains the implementation and experiment artifacts for the brain tumor segmentation project.

The main components include:

- Model implementation.
- Dataset loading and preprocessing.
- Training and checkpoint management.
- Full-volume inference.
- Quantitative evaluation.
- Qualitative visualization.
- Test result summaries.

The exact file organization may evolve as the implementation is further organized.

## Dataset

This project uses the **Brain Tumor Segmentation (BraTS) 2021** dataset.

BraTS provides multimodal MRI scans and expert-annotated tumor segmentation masks for brain tumor analysis.

Dataset access and usage are subject to the dataset provider's terms and conditions.

## Limitations and Future Work

Potential directions for further development include:

- More extensive quantitative comparisons with established 3D segmentation architectures.
- Evaluation using physical voxel spacing for surface-distance metrics.
- Additional qualitative analysis across more test patients.
- Improving segmentation performance on smaller tumor subregions.
- Further investigation of model robustness and generalization.

## Acknowledgements

The author acknowledges the BraTS 2021 dataset organizers and contributors for providing the data used in this research project.

## Author

**Nasim Beigzadeh**

- GitHub: [NasimBeigzadeh](https://github.com/NasimBeigzadeh)
