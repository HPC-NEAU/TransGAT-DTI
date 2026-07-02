
# TransGAT-DTI: transformer and graph attention network for drug-target interaction prediction

## Introduction
This repository contains the PyTorch implementation of **TransGAT-DTI** framework. **TransGAT-DTI** is a deep bilinear attention network framework with adversarial domain adaptation to explicitly learn pair-wise local interactions between drugs and targets, and adapt on out-of-distribution data. It works on two-dimensional (2D) drug molecular graphs and target protein sequences to perform prediction. We introduce a transformer and graph attention based drug-target interaction prediction framework named TransGAT-DTI to predict the interaction of drug-target interaction pairs.

![TransGAT](https://github.com/HPC-NEAU/TransGAT-DTI/blob/master/image/TransGAT.png "TransGAT")
## Required Packages
The source code developed in Python 3.7 using PyTorch 1.7.1. The required python dependencies are given below. There is no additional non-standard hardware requirements.
-   Python 3.7
-   Pytorch 1.7.1
-   NumPy 1.17.3 
-   Pytorch_geometric 1.3.2
-   tqdm 4.41.1 
-   scikit-learn 0.21.3 
- dgl 0.7.1
- dgllife 0.2.8`
- prettytable>=2.2.1
- rdkit~=2021.03.2
- yacs~=0.1.8
## Datasets
The `datasets` folder contains all experimental data used in TransGAT: [BindingDB](https://www.bindingdb.org/bind/index.jsp) [1], [BioSNAP](https://github.com/kexinhuang12345/MolTrans) [2] and [Human](https://github.com/lifanchen-simm/transformerCPI) [3]. In `datasets/bindingdb` and `datasets/biosnap` folders, we have full data with two random and clustering-based splits for both in-domain and cross-domain experiments. In `datasets/human` folder, there is full data with random split for the in-domain experiment, and with cold split to alleviate ligand bias.
## Run TransGAT on Our Data 
### Hyper-parameters and experimental setttings through command line options.

The command line options of our implementation can assign all of the expeirmental setups and model hyper-parameters, which are as follows:
`````
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
parser = argparse.ArgumentParser(description="Trans_GAT for DTI prediction")  
parser.add_argument('--cfg', required=True, help="path to config file", type=str)  
parser.add_argument('--data', required=True, type=str, metavar='TASK',  
                    help='dataset')  
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])  
args = parser.parse_args()
`````
To train TransGAT, where we provide the basic configurations for all hyperparameters in `config.py`. For different  tasks, the customized task configurations can be found in respective `configs/*.yaml` files.

For the experiments with vanilla TransGAT, you can directly run the following command. `${dataset}` could either be `bindingdb`, `biosnap` and `human`. `${split_task}` could be `random` and `cold`.
```
$ python main.py --cfg "configs/TransGAT.yaml" --data ${dataset} --split ${split_task}
```
for example：
```
$ python main.py --cfg "configs/TransGAT.yaml" --data bindingdb --split random
```
## Acknowledgements
This implementation is completed by the high-performance computing team of Northeast Agricultural University.
## Citation
If you find our code is helpful for you, feel free to cite
### Contact us

[zhouchangjian@neau.edu.cn](mailto:zhouchangjian@neau.edu.cn)
