# Robust Multiview Multimodal Driver Monitoring System Using Masked Multi-Head Self-Attention

This is the official GitHub Repo for the paper *Robust Multiview Multimodal Driver Monitoring System Using Masked Multi-Head Self-Attention* accepted by the MULA workshop at CVPR 2023.

TODO:
- [x] Upload our manually annotated labels.
- [x] Upload the code of MHSA.
- [x] Upload the code of SuMoCo.
- [x] Upload the code for training and evaluation.

## STEP1: Access the DAD dataset.

1. Visit the [official DAD repo](https://github.com/okankop/Driver-Anomaly-Detection).
2. Find the link for data downloading.
3. Download the data.
4. Unzip the data into the folder `data` (create if not exists) under this repo.

## STEP2: Prepare the dataset.

1. Copy and paste `label.csv` into the folder `data`.
2. Run `python preprocess.py` to generate the pickle files for the training set and the test set. You can specify the arguments to create your own dataset.

## STEP3: Train the model.

- Run `train.py` to train the model. You can specify the arguments, e.g., which data sources or fusion methods to use...

