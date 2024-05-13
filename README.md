# A Progressive-learning-based Channel Prediction within Phase Noise of Independent Oscillators
## Overview
This is a repository for A Progressive-learning-based Channel Prediction within Phase Noise of Independent Oscillators [Paper](https://ieeexplore.ieee.org/document/10497114) on IEEE Wireless Communications Letters. 

#### A. Abstract
The outdated downlink channel state information (CSI) contaminated by phase noise (PN) from independent local oscillators (LOs) poses a significant challenge to practical Massive Multiple-Input Multiple-Output (MIMO) Orthogonal Frequency Division Multiplexing (OFDM) systems. In this letter, we introduce a novel progressive-learning-based downlink channel prediction scheme incorporating PN compensation as a prerequisite for rebuilding the temporal correlation for channel coefficient predictions. The proposed approach is evaluated under various user equipment (UE) mobility scenarios, PN contamination
levels, and CSI delay settings, demonstrating superior predictive performance and link-level efficacy. These results position it as a viable solution for real-world applications.
#### B. Proposed Modules
Once the manuscript is ready for early access, more details about this work will be provided.

## Requirements
No special requirements
- Python >= 3.7
- For the pytorch version, we used 1.10.2+cu102, yet it is not mandatory.


## Project Details
#### A. Experiment Setup

The efficacy of the proposed progressive-learning-based framework is substantiated through comprehensive experimentation encompassing two UE mobility and CSI processing delay settings under two PN contamination levels. The dataset generation setup is presented in the following table.
<p align="center">
  <img src = "https://github.com/TeleRagingFires/Progressive/blob/8672a90f3384fa7373b9f4b89a13dd5506888e7d/Data.jpg" width="500">
</p>

## Results and Reproduction

#### Evaluation Results
The Table summarizes the overall performance with common prediction quality matrices composing the mean square error (MSE), the mean absolute error (MAE), the structure similarity (SSIM), and the cosine similarity rho.

To further verify the progressive learning-based framework and its impact on the system, a singular value decomposition (SVD) baseband precoding is conducted to simulate the average spectral efficiency R in a given receiver noise power.
![alt text](https://github.com/TeleRagingFires/Progressive/blob/139c7807a5f770da2aab967a940c2844bb750d52/Result.jpg)

## Acknowledgment
Once the manuscript is ready for early access, more details about this work will be provided.
