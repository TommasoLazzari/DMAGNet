# DMAGNet: an Interpretable Convolutional Neural Network for Galaxy Morphology Classification using Deep Dream

## Project Overview

This project develops **DMAGNet**, a convolutional neural network designed for **galaxy morphology classification** on the **Galaxy10 DECaLS** dataset.

The goal of the project is twofold:

1. Classify galaxy images into ten morphological classes using a custom deep learning architecture.
2. Investigate the internal representations learned by the network through **Google Deep Dream** visualizations.

---

## Project Structure

```text
DMAGNet/
│
├── code/
│   ├── DMAGNet_model.py
│   └── DMAGNet_deepdream.py
│
├── data/
│   ├── raw/
│   │   └── Galaxy10_DECals.h5
│   └── processed/
│       ├── train/
│       └── test/
│
├── input/
│
├── models/
│   └── DMAGNet_model23HR_Finale_ft.pt
│
├── outputs/
│
├── reports/
│   └── DMAGNet_Report.pdf
│
└── README.md
```

---

## How to Run

To reproduce the training pipeline:

1. Clone this repository.

2. Download the Galaxy10 DECaLS dataset from:

```text
https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5
```

3. Place the downloaded file in:

```text
data/raw/Galaxy10_DECals.h5
```

4. Install the required Python packages:

```bash
pip install torch torchvision numpy matplotlib opencv-python scikit-learn h5py pillow
```

5. Run the training script:

```bash
python code/DMAGNet_model.py
```

The script will prepare the dataset, train the model, fine-tune it, evaluate it on the test set, and save the trained weights in the `models/` folder.

---

## Deep Dream Visualizations

To generate Deep Dream visualizations:

1. Place one or more galaxy images inside the `input/` folder.

2. Make sure the fine-tuned model weights are available at:

```text
models/DMAGNet_model23HR_Finale_ft.pt
```

3. Run:

```bash
python code/DMAGNet_deepdream.py
```

The script will generate Deep Dream images for each input image and for each selected DMAGNet layer. The results will be saved inside the `outputs/` folder.

---

## References

- Galaxy10 DECaLS Dataset: https://zenodo.org/records/10845026
- Leung, H. & Bovy, J. Galaxy10 DECaLS: A labeled galaxy image dataset.
- Lintott et al. Galaxy Zoo: Morphologies derived from visual inspection of galaxies from the Sloan Digital Sky Survey.
- Willett et al. Galaxy Zoo 2: Detailed morphological classifications for galaxies from the Sloan Digital Sky Survey.
- Mordvintsev, A., Olah, C., & Tyka, M. Inceptionism: Going Deeper into Neural Networks.

---

## Authors

- Tommaso Lazzari
- Giovanni Chessa
- Alessandro Murru
