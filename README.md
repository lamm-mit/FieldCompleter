# FieldCompleter

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
- Example dataset: Stress field (&sigma;<sub>11</sub>) in the digital composites with linear elasticity
- The dataset can be found in the following link: https://www.dropbox.com/sh/6zkcrw2xzjtjugc/AACWo-znV2ntQC-zcvPc3KDea?dl=0

**Training**
- We train the model from a pretrained model on Places2 dataset (http://places2.csail.mit.edu/download.html)
- The pretrained checkpoint can be found here: https://www.dropbox.com/sh/eiy0n6xjc0e2a05/AADvGvn75n0WObEqBFwEletQa?dl=0
- The hyperparameters and training details can be modified via the configuration file *train-S11-pretrained.yaml*
```
python3 -u train.py --config configs/train-S11-pretrained.yaml
```

**Testing**
- The testing part is stored in *test.ipynb* including 2D field completion and inverse translation from field to geometry. 
- The pretrained checkpoints for DeepFill model and CNN model can be found here: https://www.dropbox.com/sh/1d37uqr0nj73ky9/AADMBbRw8iZgLKy2o4fJlfW3a?dl=0

