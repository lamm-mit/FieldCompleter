# FieldCompleter

- Z. Yang, M.J. Buehler, “Fill in the Blank: Transferrable Deep Learning Approaches to Recover Missing Physical Field Information,” Adv. Materials, https://doi.org/10.1002/adma.202301449, 2023
![Overall workflow](https://github.com/lamm-mit/FieldCompleter/blob/main/IMAGE_github.png)

## 2D Mechanical Field Completion using DeepFill Model
**Working directory**
```
2D_field_completer
```
**Requirements**
```
conda env create -f environment.yml
```
**Dataset**
- Example dataset: Stress field (&sigma;<sub>11</sub>) in the 2D digital composites with linear elasticity under uniaxial tension.
- The dataset can be found in the following link: https://www.dropbox.com/sh/6zkcrw2xzjtjugc/AACWo-znV2ntQC-zcvPc3KDea?dl=0.

**Training**
- We use transfer learning starting from a pretrained model trained on Places2 dataset (http://places2.csail.mit.edu/download.html).
- The pretrained checkpoint can be found here: https://www.dropbox.com/sh/eiy0n6xjc0e2a05/AADvGvn75n0WObEqBFwEletQa?dl=0.
- The hyperparameters and training details can be modified via the configuration file **train-S11-pretrained.yaml**.
```
python3 train.py --config configs/train-S11-pretrained.yaml
```

**Testing**
- The testing part is stored in **test.ipynb** including 2D field completion and inverse translation from field to geometry. 
- The pretrained checkpoints for DeepFill model and CNN model can be found here: https://www.dropbox.com/sh/1d37uqr0nj73ky9/AADMBbRw8iZgLKy2o4fJlfW3a?dl=0. The paths to checkpoints need to be specified in **test.ipynb**.

## 3D Mechanical Field Completion using modified ViViT model
**Working directory**
```
3D_field_completer
```
**Requirements**
```
conda env create -f environment.yml
```
**Dataset**
- Example dataset: Stress field (&sigma;<sub>11</sub>) in the 3D digital composites with linear elasticity under uniaxial compression.
- The dataset can be found in the following link: https://www.dropbox.com/sh/5gntfr7ittue5fh/AACE2D-GOeTHhR2zCMcUCXila?dl=0. **S11.npy** store matrix represent all 3D stress fields. **labels_train.npy** and **labels_test.npy** are train/test sequences representing geometries of 3D composites. 

**Training**
- The training starts from scratch.
- The hyperparameters and training details can be modified directly in **vivit.py**.
```
python3 train.py 
```

**Testing**
- The testing part is stored in **test.ipynb** including 3D field completion and inverse translation from field to geometry. 
- The pretrained checkpoints for ViViT model and CNN model can be found here: https://www.dropbox.com/sh/ulz37l3ang5hfjf/AAB1dr2yX2AJw26bGSE582S4a?dl=0. The paths to checkpoints need to be specified in **test.ipynb**.
