#  Burned Areas Delineation
In this repository, it is shared the source code used in the Master Thesis "Delineation of Burned Areas on SPOT 6/7 Imagery for Emergency Management"

## About
After a wildfire, delineating burned areas is crucial to quantify damages and support informed recovery and remediation strategies for affected ecosystems and populations. Current approaches for mapping burned areas mainly rely on computer vision models trained with post-event remote sensing imagery. However, these models typically focus on improving delineation performance and often overlook their applicability to time-constrained emergency management scenarios. 

This study presents a supervised semantic segmentation workflow designed to enhance both the performance and efficiency of burned area delineation. All the conducted experiments are evaluated using standard segmentation metrics such as Dice Score (Dice) and Intersection Over Union (IoU), along with inference time.

## Main Steps Methodology
 It consists of three main stages: Training, Testing, and Prediction. In each stage different techniques are evaluated.

**Figure Abreviations:**
* STF - MTF: Single and Multi Task Frameworks
* NDVI, NDWI, BAI: Spectral Indices
* CM: Cloud Mask
* TTA: Test-Time Augmentation
* MP: Mixed Precision
* FP: Floating Point


<img src="https://github.com/mariarodriguezn/BurnedAreasDelineation/blob/main/docs/Methodology.png" alt="Methodology" width="700">

## References
Arnaudo, E., Barco, L., Merlo, M., & Rossi, C. (2023). Robust Burned Area Delineation 
through Multitask Learning.
