# DMAGNet: an Interpretable Convolutional Neural Network for Galaxy Morphology Classification using Deep Dream

---

## Project Overview

This project presents **DMAGNet**, a convolutional neural network designed for **galaxy morphology classification** with a strong emphasis on **model interpretability**.

The work combines modern deep learning techniques with visualization-based interpretability methods to address two key objectives:

1. Achieve reliable automatic classification of galaxy morphologies from astronomical images  
2. Provide qualitative insights into the internal representations learned by the network  

To this end, we integrate **multi-scale convolutional design**, **attention mechanisms**, and the **Google Deep Dream algorithm** to visually inspect how different network layers respond to galaxy structures.

---

## Dataset

The experiments are conducted on the **Galaxy10 DECaLS** dataset, which contains approximately **18,000 RGB galaxy images** grouped into **10 morphological classes**.

Key characteristics:

- Derived from **Galaxy Zoo** visual annotations  
- Based on **DECaLS survey imaging**, offering improved spatial resolution  
- Covers a wide range of galaxy morphologies, including smooth, spiral, barred, merging, and edge-on systems  

All images are kept at their original resolution of **256 × 256 pixels**.

---

## Methods and Techniques

The project integrates architectural design, supervised learning, and interpretability analysis:

- **DMAGNet Architecture**
  - Hierarchical convolutional structure with progressive spatial abstraction
  - **DMA blocks** for multi-scale feature extraction
  - **Local Attention Module (LAM)** for spatially adaptive feature refinement
  - Lightweight classification head for final prediction

- **Training and Evaluation**
  - Supervised classification on 10 morphological classes
  - Standard data augmentation techniques
  - Evaluation using accuracy and F1-score metrics

- **Interpretability**
  - Application of **Google Deep Dream** to multiple network layers
  - Layer-wise visualization of learned features
  - Multi-scale dreaming and **multi-resolution re-dreaming pipeline**
  - Qualitative analysis of how representations evolve with depth

---

## Files Included

*(To be completed)*

---

## How to Run

*(To be completed)*

---

## Results and Interpretability

The model achieves competitive classification performance on the Galaxy10 DECaLS test set while maintaining interpretability.

Deep Dream visualizations reveal a clear hierarchical organization of learned features:
- Early layers respond to local photometric patterns
- Intermediate layers capture structured morphological motifs
- The final attention module focuses on globally coherent galaxy structures

These observations suggest that DMAGNet encodes **physically meaningful representations** aligned with established astronomical morphology concepts.

---

## References

- Hubble, E. P. (1926). *Extragalactic nebulae*. The Astrophysical Journal.  
- Hubble, E. P. (1936). *The Realm of the Nebulae*. Yale University Press.  
- de Vaucouleurs, G. (1963). *Classification and morphology of external galaxies*. ApJS.  
- Lintott, C. J. et al. (2008). *Galaxy Zoo*. MNRAS.  
- Willett, K. W. et al. (2013). *Galaxy Zoo 2*. MNRAS.  
- Mordvintsev, A., Olah, C., & Tyka, M. (2015). *Inceptionism: Going Deeper into Neural Networks*. Google Research Blog.
