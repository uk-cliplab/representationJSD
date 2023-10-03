# The representation Jensen Shannon divergence

This repository is the official implementation of the representation JS divergence and all the experiments in the article. Namely, Neural estimation of Jensen-Shannon divergence of Cauchy distributions, mode collapse in generative adversarial networks (GANs), GAN on the stacked MNIST dataset, and two sample testing. 

### Abstract: 
_Statistical divergences quantify the difference between probability distributions, thereby allowing for multiple uses in machine-learning. However, a fundamental challenge of these quantities is their estimation from empirical samples since the underlying distributions of the data are usually unknown. In this work, we propose a divergence inspired by the Jensen-Shannon divergence which avoids the estimation of the probability density functions. Our approach embeds the data in an reproducing kernel Hilbert space (RKHS) where we associate data distributions with uncentered covariance operators in this representation space. Therefore, we name this measure the representation Jensen-Shannon divergence (RJSD). We provide an estimator from empirical covariance matrices by explicitly mapping the data to an RKHS using Fourier features. This estimator is flexible, scalable, differentiable, and suitable for minibatch-based optimization problems. Additionally, we provide an estimator based on kernel matrices without an explicit mapping to the RKHS. We provide consistency convergence results for the proposed estimator. Moreover, we demonstrate that this quantity is a lower bound on the Jensen-Shannon divergence, leading to a variational approach to estimate it with theoretical guarantees. We leverage the proposed divergence to train generative networks, where our method mitigates mode collapse and encourages samples diversity.  Additionally, RJSD surpasses other state-of-the-art techniques in multiple two-sample testing problems, demonstrating superior performance and reliability in discriminating between distributions._


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
 Additionally, it is necessary to install the representation Information theoretic learning library. 
 
1) Move to folder:  ```cd representation-itl```

2) Install with pip:  ```pip install -e .```

## Usage

- The neural estimation of JS divergence and the baselines can be found in the main_neural_Estimators.ipynb file. 

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

![figure1](https://github.com/uk-cliplab/representationJSD/assets/84861891/7b635d25-4b12-4111-ae73-8c0e5d3ae519)

![figure2](https://github.com/uk-cliplab/representationJSD/assets/84861891/27a6190e-6b9b-4b00-8492-faeb37e9a328)
<p align="center">
  <img src="https://github.com/uk-cliplab/representationJSD/assets/84861891/66669887-6a94-4b9c-97a6-6340e0253c97" width="420" height="350">
</p>


