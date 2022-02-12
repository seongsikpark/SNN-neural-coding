# Readme

Deep SNNs with various neural coding methods

Spiking Neural Network library based on TensorFlow (V2)

DNN-to-SNN conversion based 


# Neural Coding






# How to run?
Please refer to provided shell script (run.sh).

## DNN-to-SNN conversion
1. Run DNN inference w/ fused batchnorm. and to collect activation statistics.
  - related configurations
```
in run.sh
nn_mode=ANN

in ./configs/weight_norm.conf
f_write_stat_train_data=True
```
2. Run SNN w/ data-based norm.
```
in run.sh
nn_mode=SNN

in ./configs/weight_norm.conf
f_write_stat_train_data=False
f_fused_bn=True
```

## Set neural coding 
```
in run.sh

input_spike_mode={'REAL',POISSON',WEIGHTED_SPIKE','BURST','TEMPORAL'}
neural_coding={POISSON',WEIGHTED_SPIKE','BURST','TEMPORAL'}
```

## Gradient-based optimization of temporal kernel parameters (T2FSNN, DAC-20)
1. Set ```nn_mode=SNN```, ```f_train_time_const=True``` in run.sh
2. Set the number of train epoch ```epoch_train_time_const``` and save interval ```time_const_save_interval``
   
   (Total numver of train data = (the number of train epoch) x (save interval))

3. Run and Train
4. Set ```f_train_time_const=False```, ```f_load_time_const=True```, and ```time_const_num_trained_data=# of trained data```
5. Run and Inference

(if ```f_train_time_const=True``` and ```f_load_time_const=True```, load and train kernel parameters)


## Early firing (T2FSNN, DAC-20)
- Conventional method
```time_fire_start == time_fire_duration```
- Early firing
```time_fire_start < time_fire_duration```
  (e.g.,```time_fire_start=40,time_fire_duration=80```)
  

# Download models
https://www.dropbox.com/sh/6ubl8y3s8jdpj6v/AACf0IIcNhYnPUGDn8ELKKRja?dl=0

(models should be unzipped and located in ```./models_ckpt``` directory)


# Publications
Fast and Efficient Information Transmission in Deep Spiking Neural Networks (DAC-19)
(https://dl.acm.org/citation.cfm?id=3316781.3317822)

T2FSNN: Deep Spiking Neural Networks with Time-to-first-spike Coding (DAC-20)
(https://dl.acm.org/doi/10.5555/3437539.3437564)
