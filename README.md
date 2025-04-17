# Unsupervised Sequence Detection in Capsule Endoscopic Videos

| Patient 4ab | Patient 4ae |
|-------------|-------------|
| ![Example 4ab](assets/Examples/4ab.svg) | ![Example 4ae](assets/Examples/4ae.svg) |


This repository contains the code and resources for my Bachelor's thesis titled **"Unsupervised Sequence Detection in Capsule Endoscopic Video,"** submitted at TU Darmstadt in December 2024.

## Overview
Wireless Capsule Endoscopy (WCE) provides a non-invasive method to inspect the gastrointestinal (GI) tract, particularly useful in diagnosing diseases such as Crohn’s disease. However, reviewing the lengthy video footage is highly time-consuming. This thesis proposes a methodology to summarize capsule endoscopic videos, significantly reducing the redundant information that medical professionals must review.

The complete thesis can be downloaded [here](https://github.com/bapierro/WCE-Sequences-Thesis/blob/main/Bachelorthesis_Pierre_Bathoum.pdf).


## Objectives
- To reduce the evaluation burden of capsule endoscopy videos by summarizing content into meaningful segments.
- Leverage unsupervised and self-supervised machine learning methods due to the limited availability of annotated data.
- Provide a tool for automatically segmenting videos into semantically coherent chapters to assist medical diagnostics.

## Methodology
### Feature Extraction
Utilizes a Vision Transformer (ViT-B/16) architecture trained via self-supervised learning (DINOv1) to extract expressive, informative features from video frames.

### Temporal Smoothing
Applies Gaussian temporal smoothing to the extracted features to reduce noise and emphasize significant transitions within the video data.

### Hierarchical Density-Based Clustering
Uses HDBSCAN, a hierarchical density-based clustering approach, to group frames into clusters. We identify segments with high redundancy to group frames into clusters, identifying segments with high redundancy and summarizing the video.

The idea is that similar frames lead to similar features in the latent space, therefore temporal redundancy, meaning redundant frames in the video, can be detected by different densities in our latent space.

## Datasets
- **Kvasir-Capsule:** Provides annotated frames and videos from capsule endoscopic videos.
- **SeeAI:** High-resolution annotated frames covering many gastrointestinal conditions.
- **KID:** Annotated images of various gastrointestinal abnormalities.

## Results
The thesis tries to leverage unsupervised and self-supervised methods to reduce the video length and redundancy significantly, with the vision to enable quicker diagnostic reviews while maintaining essential diagnostic content.

### Example Visualizations

The methodology illustrated in my thesis enables us to create summaries of WCE videos like in the following (all examples use videos taken out of Kvasir-Capsule):

| Patient 4ff | Patient 5a5 |
|-------------|-------------|
| ![Example 4ff](assets/Examples/4ff.svg) | ![Example 5a5](assets/Examples/5a5.svg) |


## Archive
This repository is archived and read-only, as it represents the final version of my thesis work and will not receive further updates.

## Author
Pierre Emmanuel Lou Bathoum  
TU Darmstadt, Fraunhofer IGD (MECLab)

## Advisors
- Ph.D. Anirban Mukhopadhyay  
- M.Sc. Henry John Krumb

## Reference this work

If you find the ideas or methods in this thesis useful for your own projects, you can reference it like this:

`Bathoum, Pierre Emmanuel Lou. “Unsupervised Sequence Detection in Capsule Endoscopic Video.” Bachelor Thesis, TU Darmstadt, December 2024. https://github.com/bapierro/WCE-Sequences-Thesis`


----


