## Distributional Sliced Wasserstein distance
This is a pytorch implementation of distributional sliced Wasserstein which is a sliced optimal transport distance between two probability measures. Details of the model architecture and experimental results can be found in: "[our following paper](https://arxiv.org/pdf/2002.07367.pdf)".

### Bibtex:
````
@inproceedings{
nguyen2021distributional,
title={Distributional Sliced-Wasserstein and Applications to Generative Modeling},
author={Khai Nguyen and Nhat Ho and Tung Pham and Hung Bui},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=QYjO70ACDK}
}
````
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.
### Installation

* python3.6
* pytorch 1.4
* torchvision
* numpy
* tqdm
* POT

````
pip install torch==1.4.0 torchvision==0.5.0 numpy tqdm POT
````
### Train on MNIST
``` 
python mnist.py \
    --datadir='./' \
    --outdir='./result' \
    --batch-size=512 \
    --seed=16 \
    --p=2 \
    --lr=0.0005 \
    --dataset='MNIST'
    --model-type='DSWD'\
    --latent-size=32 \ 
```
```
model-type in (SWD|MSWD|DSWD|GSWD|DGSWD|JSWD|JMSWD|JDSWD|JGSWD|JDGSWD|MGSWNN|JMGSWNN|MGSWD|JMGSWD)
```
Options for Sliced distances (number of projections used to approximate the distances)
````
--num-projection=1000
````

Options for Max Sliced-Wasserstein distance and Distributional distances (number of gradient steps for find the max slice or the optimal push-forward function):

```
--niter=10
```
Options for Distributional Sliced-Wasserstein Distance and Distributional Generalized Sliced-Wasserstein Distance (regularization strength)

````
--lam=10
````
Options for Generalized Wasserstein Distance (using circular function for Generalized Radon Transform)

````
--r=1000;\
--g='circular'
````
### Train on CELEBA and CIFAR10 and LSUN
``` 
python main.py \
    --datadir='./' \
    --outdir='./result' \
    --batch-size=512 \
    --seed=16 \
    --p=2 \
    --lr=0.0005 \
    --model-type='DSWD'\
    --dataset='CELEBA'
    --latent-size=100 \ 
```
```
model-type in (SWD|MSWD|DSWD|GSWD|DGSWD)
```
Options for Sliced distances (number of projections used to approximate the distances)
````
--num-projection=1000
````
Options for Max Sliced-Wasserstein distance and Distributional distances (number of gradient steps for find the max slice or the optimal push-forward function):

```
--niter=1
```
Options for Distributional Sliced-Wasserstein Distance and Distributional Generalized Sliced-Wasserstein Distance (regularization strength)

````
--lam=1
````
Options for Generalized Wasserstein Distance (using circular function for Generalized Radon Transform)

````
--r=1000;\
--g='circular'
````
### Evaluation
Please use https://github.com/bioinf-jku/TTUR for evaluating the trained generative models.

## Some generated images
##### MNIST generated images
![MNIST](images/MNIST.png) 
##### CELEBA generated images
![MNIST](images/CelebA.png) 
##### LSUN generated images
![MNIST](images/LSUN.png) 
