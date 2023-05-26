# The representation Jensen Shannon divergence

This repository is the official implementation of the representation JS divergence and all the experiments in the article. Namely, Neural estimation of Jensen-Shannon divergence of Cauchy distributions, mode collapse in generative adversarial networks (GANs), GAN on the stacked MNIST dataset, and two sample testing. 

### Abstract: 
_Statistical divergences quantify the difference between probability distributions finding multiple uses in machine-learning. However, a fundamental challenge is to estimate divergence from empirical samples since the underlying distributions of the data are usually unknown. In this work, we propose the representation Jensen-Shannon Divergence, a novel divergence based on covariance operators in reproducing kernel Hilbert spaces (RKHS). Our approach embeds the data distributions in an RKHS and exploits the spectrum of the covariance operators of the representations. We provide an estimator from empirical covariance matrices by explicitly mapping the data to an RKHS using Fourier features. This estimator is flexible, scalable, differentiable, and suitable for minibatch-based optimization problems. Additionally, we provide an estimator based on kernel matrices without having an explicit mapping to the RKHS. We show that this quantity is a lower bound on the Jensen-Shannon divergence, and we propose a variational approach to estimate it. We applied our divergence to two-sample testing outperforming related state-of-the-art techniques in several datasets. We used the representation Jensen-Shannon divergence as a cost function to train generative adversarial networks which intrinsically avoids mode collapse and encourages diversity._


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
 Additionally, it is necessary to install the representation Information theoretic learning library. 
 
1) Move to folder:  ```cd representation-itl```

2) Install with pip:  ```pip install -e .```

## Usage

- The neural estimation of JS divergence and the baselines can be found in the JSDcauchydensities.ipynb file. 

- To reproduce the mode collapse results in the paper, run this command:

```train
python3 main.py --loss standard --lr_d 0.0005 --lr_g 0.0001 --noise_type uniform --plot_frequency 100
```
- The GAN experiment with all the configurations used in the paper, and the evaluations are in the main.ipynb in the GAN_stacked_MNIST folder. 
- To reproduce the two sample testing experiments run the following commands. The code is set up to run in parallel with different processors running indepent repetitions of the algorithm. To run it in a single CPU, change the parallel option to False in the run_experiment_ function. 

```train
python3 python3 -u TSTblobs.py \
        -e 'blobs' \
        -nPerm 100 \
        -nTest 100 \
        -sign 0.05 \
        -rId REPETITIONID \
        -datafolder DATAFOLDER \
```

## Results

The achieved results in each of the experiments can be seen in the next figures:


![Screenshot from 2023-05-24 10-56-14](https://github.com/uk-cliplab/representationJSD/assets/84861891/27065b18-2af9-4be2-94c1-9b7d62c6c0d4)


![Screenshot from 2023-05-24 10-56-46](https://github.com/uk-cliplab/representationJSD/assets/84861891/27a6190e-6b9b-4b00-8492-faeb37e9a328)


| Model name         | Number of modes |  KL divergence |
| ------------------ |---------------- | -------------- |
| Representation JSD |     1000        |     0.04       |

