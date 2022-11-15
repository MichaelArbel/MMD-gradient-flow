## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
   * [Student Teacher network](#student-teacher-network)
* [Resources](#resources)
   * [Hardware](#hardware)
* [Full documentation](#full-documentation)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the Wasserstein gradient flow of the Maximum Mean Discrepancy from [Maxmimum Mean Discrepancy Gradient Flow paper](https://arxiv.org/abs/1906.04370) published at Neurips 2019. It allows to reproduce the experiments in the paper.


## Requirements


This a Pytorch implementation which requires the follwoing packages:

```
python==3.6.2 or newer
torch==1.2.0 or newer
torchvision==0.4.0 or newer
numpy==1.17.2  or newer
```

All dependencies can be installed using:

```
pip install -r requirements.txt
```




## How to use

### Student Teacher network:
```
python train_student_teacher.py --device=-1 
```

## Resources

### Hardware

To use a particular GPU, set —device=#gpu_id
To use GPU without specifying a particular one, set —device=-1
To use CPU set —device=-2


## Full documentation

```
# Optimizer parameters 
--lr 					learning rate [1.]
--batch_size			batch size [100]
--total_epochs			total number of epochs [10000]
--Optimizer 			Optimizer ['SGD']
--use_scheduler 		By default uses the ReduceLROnPlateau scheduler [False]

# Loss paramters
--loss					loss to optimize: mmd_noise_injection, mmd_diffusion, sobolev ['mmd_noise_injection']
--with_noise		  	to use noise injection set to true [True]
 --noise_level			variance of the injected noise [1.]
--noise_decay_freq		decays the variance of the injected every 1000 epochs by a factor "noise_decay" [1000]
--noise_decay 			factor for decreasing the variance of the injected noise [0.5]

# Hardware parameters 
--device				gpu device, set -1 for cpu [0]
--dtype					precision: single: float32 or double: float64 ['float32']

# Reproducibility parameters  
--seed					seed for the random number generator on pytorch [1]
--log_dir				log directory ['']
--log_name				log name ['mmd']
--log_in_file			to log output on a file [False]


--bias					ste to include bias in the network parameters [False]
--teacher_net			teacher network ['OneHidden']
--student_net				student network ['NoisyOneHidden']
--d_int					dim input data [50]
--d_out					dim out feature [1]
--H 					num of hidden layers in the teacher network [3]
--num_particles 		num_particles*H = number of hidden units in the student network [1000]

# Initialization parameters
--mean_student 			mean initial value for the student weights [0.001]
--std_student 			std initial value for the student weights [1.]
--mean_teacher 			mean initial value for the teacher weights [0.]
--std_teacher			std initial value for the teacher weights [1.]

# Data parameters 
--input_data 			input data distribution ['Spherical']
--N_train				num samples for training [1000]
--N_valid				num samples for validation [1000]

--config				config file for non default parameters ['']

```

## Reference

If using this code for research purposes, please cite:

[1] M. Arbel, A. Korba, A. Salim, A. Gretton [*Maximum Mean Discrepancy Gradient Flow*](https://arxiv.org/abs/1906.04370)

```
@article{Arbel:2018,
			author  = {Michael Arbel, Anna Korba, Adil Salim, Arthur Gretton},
			title   = {Maximum Mean Discrepancy Gradient Flow},
			journal = {NeurIPS},
			year    = {2019},
			url     = {https://arxiv.org/abs/1906.04370},
}
```


## License 

This code is under a BSD license.
