#!/bin/bash

#
source ./venv/bin/activate


###############################################################################
## Include
###############################################################################
# path
source ./configs/path.conf

# weight normalization
source ./configs/weight_norm.conf

# utils
source ./configs/utils.conf

# model description
source ./configs/models_descript.conf


###############################################################################
## Model & Dataset
###############################################################################
#nn_mode='ANN'
nn_mode='SNN'


#exp_case='CNN_MNIST'
exp_case='VGG16_CIFAR-10'
#exp_case='VGG16_CIFAR-100

###############################################################################
## Run
###############################################################################
#training_mode=True
training_mode=False

#
# If this flag is False, then the trained model is overwritten
load_and_train=False
#load_and_train=True

#
regularizer='L2'
#regularizer='L1'


# full test
f_full_test=True
#f_full_test=False

# valid when "TEMPORAL" coding is not used
time_step=1000

#
time_step_save_interval=10


###############################################################
# Batch size - small test
###############################################################

# only when (f_full_test = False)
batch_size=25

idx_test_dataset_s=0
num_test_dataset=25


###############################################################################
## Neural coding
###############################################################################

#
## input spike mode
#
#input_spike_mode='REAL'
#input_spike_mode='POISSON'
#input_spike_mode='WEIGHTED_SPIKE'
#input_spike_mode='BURST'
input_spike_mode='TEMPORAL'

#
## neural coding
#
#neural_coding='RATE'
#neural_coding='WEIGHTED_SPIKE'
#neural_coding='BURST'
neural_coding='TEMPORAL'


###############################################################################
## Neuron
###############################################################################

# default: IF
#n_type='LIF'
n_type='IF'


# refractory mode
#f_refractory=True
f_refractory=False


###############################################################
# for weighted spike coding (phase coding)
###############################################################
p_ws=8

###############################################################
# initial thresholds
###############################################################

vth_n_in=1.0

if [ ${input_spike_mode} = 'BURST' ]
then
    vth_n_in=0.125
fi


vth=1.0        # weight norm. default
# TODO: input_spike_mode - REAL
if [ ${neural_coding} = 'RATE' ]
then
    vth=0.4
elif [ ${neural_coding} = 'BURST' ]
then
    vth=0.4         # empirical best
elif [ ${neural_coding} = 'TEMPORAL' ]
then
    #vth=0.8
    vth=1.0
fi



###############################################################################
## SNN output type
###############################################################################

#snn_output_type='SPIKE'
snn_output_type='VMEM'
#snn_output_type='FIRST_SPIKE_TIME'       # for TTFS coding


###############################################################################
## Gradient-based optimization of tc and td
## only for the TTFS coding (TEMPORAL)
## "Deep Spiking Neural Networks with Time-to-first-spike Coding", DAC-20
###############################################################################

f_visual_record_first_spike_time=False
#f_visual_record_first_spike_time=True

#
#f_load_time_const=False
f_load_time_const=True

#
# train time constant for temporal coding
f_train_time_const=False
#f_train_time_const=True

#
f_train_time_const_outlier=True
#f_train_time_const_outlier=False


#
time_const_init_file_name='./temporal_coding'

time_const_num_trained_data=60000
#time_const_num_trained_data=50000
#time_const_num_trained_data=40000
#time_const_num_trained_data=30000
#time_const_num_trained_data=20000
#time_const_num_trained_data=10000
#time_const_num_trained_data=0

#
#time_const_save_interval=50000
time_const_save_interval=10000
#time_const_save_interval=0

#
epoch_train_time_const=6

# TTFS - MNIST default setting
#tc=5
#time_fire_start=20    # integration duration
#time_fire_duration=20   # time window
#time_window=${time_fire_duration}

tc=20
time_fire_start=80   # integration duration
#time_fire_start=40   # integration duration
time_fire_duration=80   # time window
time_window=${time_fire_duration}

###############################################################################


num_epoch=1000
save_interval=50

lr=0.001
lr_decay=0.1
lr_decay_step=100
lamb=0.0001
batch_size_training=128


###############################################################################
## DO NOT TOUCH
###############################################################################

if [ ${training_mode} = True ]
then
    _exp_case=TRAIN_${exp_case}
else
    _exp_case=INFER_${exp_case}
fi


if [ ${training_mode} = True ]
then
    if [ ${load_and_train} = True ]
    then
        en_load_model=True
    else
        en_load_model=False
    fi

    en_train=True

    #nn_mode='ANN'
    f_fused_bn=False
    f_stat_train_mode=False
    f_w_norm_data=False
    # TODO: training and inference bach size (training / validation)
    batch_size=${batch_size_training}
    f_full_test=True

else
    en_train=False
    en_load_model=True
fi


case ${_exp_case} in
###############################################################
## Inference setup
###############################################################
INFER_CNN_MNIST)
    echo "Inference mode - "${nn_mode}", Model: CNN, Dataset: MNIST"
    dataset='MNIST'
    ann_model='CNN'
    #model_name='cnn_mnist_ro_0'
    model_name='cnn_mnist_train_ANN'

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        time_step="$((4*${time_fire_start} + ${time_fire_duration}))"
    fi
    ;;


INFER_VGG16_CIFAR-10)
    echo "Inference mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-10"
    ann_model='VGG16'
    dataset='CIFAR-10'
    model_name='vgg_cifar_ro_0'

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        time_step="$((15*${time_fire_start} + ${time_fire_duration}))"
    fi
    ;;


INFER_VGG16_CIFAR-100)
    echo "Inference mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-100"
    ann_model='VGG16'
    dataset='CIFAR-100'
    model_name='vgg_cifar100_ro_0'
    #model_name='vgg16_cifar100_train_ANN'


    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        time_step="$((15*${time_fire_start} + ${time_fire_duration}))"
    fi
    ;;



###############################################################
## Training setup
###############################################################
TRAIN_CNN_MNIST)
    echo "Training mode - "${nn_mode}", Model: CNN, Dataset: MNIST"
    dataset='MNIST'
    ann_model='CNN'

    num_epoch=10000

    model_name='cnn_mnist_train_'${nn_mode}

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        time_step="$((4*${time_fire_start} + ${time_fire_duration}))"
    fi
    ;;

TRAIN_VGG16_CIFAR-10)
    echo "Training mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-10"
    ann_model='VGG16'
    dataset='CIFAR-10'

    model_name='vgg16_cifar10_train_'${nn_mode}

    if [ ${f_full_test} = True ]
    then
        #batch_size=400
        batch_size=250
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        time_step="$((16*${time_fire_start} + ${time_fire_duration}))"
    fi
    ;;

TRAIN_VGG16_CIFAR-100)
    echo "Training mode - "${nn_mode}", Model: VGG16, Dataset: CIFAR-100"
    ann_model='VGG16'
    dataset='CIFAR-100'

    #num_epoch=2000
    #num_epoch=4000
    num_epoch=10000

    model_name='vgg16_cifar100_train_'${nn_mode}

    if [ ${f_full_test} = True ]
    then
        batch_size=400
        idx_test_dataset_s=0
        num_test_dataset=10000
    fi

    if [ ${neural_coding} = "TEMPORAL" ]
    then
        time_step="$((16*${time_fire_start} + ${time_fire_duration}))"
    fi
    ;;

*)
    echo "not supported experiment case:" ${_exp_case}
    exit 1
    ;;
esac







###############################################
##
###############################################
# record first spike time of each neuron - it should be True for training time constant
# overwirte this flag as True if f_train_time_const is Ture
# it should be True when TTFS coding is used
if [ ${neural_coding} = "TEMPORAL" ]
then
    f_record_first_spike_time=True
    f_refractory=True
else
    f_record_first_spike_time=False
fi



###############################################3
## training time constant
###############################################3
if [ ${f_train_time_const} = True ]
then
    f_record_first_spike_time=True
fi
###############################################3

###############################################3
## visual spike time ditribution
###############################################3
if [ ${f_visual_record_first_spike_time} = True ]
then
    f_record_first_spike_time=True

    idx_test_dataset_s=0
    num_test_dataset=${batch_size}
fi
###############################################3




#
echo "time_step: " ${time_step}


case ${dataset} in
MNIST)
    echo "Dataset: MNIST"
    num_class=10
    input_size=28
    ;;
CIFAR-10)
    echo "Dataset: CIFAR-10"
    num_class=10
    input_size=28
    ;;
CIFAR-100)
    echo "Dataset: CIFAR-100"
    num_class=100
    input_size=28
    ;;
*)
    echo "not supported dataset"
    num_class=0
    exit 1
    ;;
esac



if [ ${nn_mode} = 'ANN' ]
then
    echo "ANN mode"
    log_file_post_fix=''
else
    echo "SNN mode"
    log_file_post_fix=_${n_type}_time_step_${time_step}_vth_${vth}_c_infer
fi

log_file=${path_log_root}/log_${model_name}_${nn_mode}_${log_file_post_fix}


date=`date +%Y%m%d_%H%M`

path_result_root=${path_result_root}/${model_name}
time_const_root=${time_const_init_file_name}/${model_name}

#
path_stat=${path_stat_root}/${model_name}

#
mkdir -p ${path_log_root}
mkdir -p ${path_result_root}
mkdir -p ${time_const_root}
mkdir -p ${path_stat}



log_file=${path_log_root}/${date}.log

#{ unbuffer time kernprof -l main.py \
#{ CUDA_VISIBLE_DEVICES=0 unbuffer time python main.py \
#{ unbuffer time python -m line_profiler main.py \
{ unbuffer time python -u main.py \
    -date=${date}\
    -time_step_save_interval=${time_step_save_interval}\
    -en_load_model=${en_load_model}\
    -checkpoint_load_dir=${path_models_ckpt}\
	-checkpoint_dir=${path_models_ckpt}\
	-model_name=${model_name}\
   	-en_train=${en_train}\
	-save_interval=${save_interval}\
	-nn_mode=${nn_mode}\
	-regularizer=${regularizer}\
	-epoch=${num_epoch}\
	-idx_test_dataset_s=${idx_test_dataset_s}\
	-num_test_dataset=${num_test_dataset}\
	-n_type=${n_type}\
	-time_step=${time_step}\
	-n_init_vth=${vth}\
	-n_in_init_vth=${vth_n_in}\
	-lr=${lr}\
    -lr_decay=${lr_decay}\
    -lr_decay_step=${lr_decay_step}\
    -lamb=${lamb}\
    -ann_model=${ann_model}\
    -num_class=${num_class}\
    -f_fused_bn=${f_fused_bn}\
    -f_stat_train_mode=${f_stat_train_mode}\
    -f_vth_conp=${f_vth_conp}\
    -f_spike_max_pool=${f_spike_max_pool}\
    -f_w_norm_data=${f_w_norm_data}\
    -p_ws=${p_ws}\
    -f_refractory=${f_refractory}\
    -input_spike_mode=${input_spike_mode}\
    -neural_coding=${neural_coding}\
    -f_tot_psp=${f_tot_psp}\
    -f_write_stat=${f_write_stat}\
    -act_save_mode=${act_save_mode}\
    -f_save_result=${f_save_result}\
    -path_stat=${path_stat}\
    -path_result_root=${path_result_root}\
    -prefix_stat=${prefix_stat}\
    -tc=${tc}\
    -time_window=${time_window}\
    -time_fire_start=${time_fire_start}\
    -time_fire_duration=${time_fire_duration}\
    -f_record_first_spike_time=${f_record_first_spike_time}\
    -f_visual_record_first_spike_time=${f_visual_record_first_spike_time}\
    -f_train_time_const=${f_train_time_const}\
    -f_train_time_const_outlier=${f_train_time_const_outlier}\
    -f_load_time_const=${f_load_time_const}\
    -time_const_init_file_name=${time_const_init_file_name}\
    -time_const_num_trained_data=${time_const_num_trained_data}\
    -time_const_save_interval=${time_const_save_interval}\
    -epoch_train_time_const=${epoch_train_time_const}\
    -snn_output_type=${snn_output_type}\
    -dataset=${dataset}\
    -input_size=${input_size}\
    -batch_size=${batch_size}\
    ; } 2>&1 | tee ${log_file}

echo 'log_file: '${log_file}
