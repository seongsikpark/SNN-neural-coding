from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0,'./')

from datetime import datetime


#en_gpu=False
en_gpu=True

gpu_number=0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

#
# 0: all messages
# 1: INFO not printed
# 2: INFO, WARNING not printed
# 3: INFO, WARNING, ERROR not printed
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

if en_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

#
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.profiler import option_builder

builder = option_builder.ProfileOptionBuilder

#
import train
import test


# models
from models import cnn_mnist
from models import VGG16


# dataset
from datasets import mnist
from datasets import cifar10
from datasets import cifar100



import shutil

#
import pprint

now = datetime.now()

#
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=4)

#
pp = pprint.PrettyPrinter().pprint

#
flags = tf.compat.v1.app.flags
tf.compat.v1.app.flags.DEFINE_string('date','','date')

tf.compat.v1.app.flags.DEFINE_integer('epoch', 300, 'Number os epochs')
tf.compat.v1.app.flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 100, '')

tf.compat.v1.app.flags.DEFINE_string('nn_mode', 'SNN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')

tf.compat.v1.app.flags.DEFINE_string('output_dir', './tensorboard', 'Directory to write TensorBoard summaries')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_dir', './models_ckpt', 'Directory to save checkpoints')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_load_dir', './models_ckpt', 'Directory to load checkpoints')
tf.compat.v1.app.flags.DEFINE_bool('en_load_model', False, 'Enable to load model')

tf.compat.v1.app.flags.DEFINE_boolean('en_train', False, 'enable training')


tf.compat.v1.app.flags.DEFINE_boolean('use_bias', True, 'use bias')

#
tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.7, 'initial value of vth')
tf.compat.v1.app.flags.DEFINE_float('n_in_init_vth', 0.7, 'initial value of vth of n_in')
tf.compat.v1.app.flags.DEFINE_float('n_init_vinit', 0.0, 'initial value of vinit')
tf.compat.v1.app.flags.DEFINE_float('n_init_vrest', 0.0, 'initial value of vrest')

# adam optimizer
tf.compat.v1.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.compat.v1.app.flags.DEFINE_float('momentum', 0.9, 'momentum')

# regularization
tf.compat.v1.app.flags.DEFINE_float('lamb',0.0001, 'lambda')

tf.compat.v1.app.flags.DEFINE_float('lr_decay', 0.1, '')
tf.compat.v1.app.flags.DEFINE_integer('lr_decay_step', 50, '')

tf.compat.v1.app.flags.DEFINE_integer('time_step', 10, 'time steps per sample in SNN')


tf.compat.v1.app.flags.DEFINE_integer('idx_test_dataset_s', 0, 'start index of test dataset')
tf.compat.v1.app.flags.DEFINE_integer('num_test_dataset', 10000, 'number of test datset')
tf.compat.v1.app.flags.DEFINE_integer('size_test_batch', 1, 'size of test batch') # not used now

tf.compat.v1.app.flags.DEFINE_integer('save_interval', 10, 'save interval of model')

tf.compat.v1.app.flags.DEFINE_bool('en_remove_output_dir', False, 'enable removing output dir')


#
tf.compat.v1.app.flags.DEFINE_string('regularizer', 'L2', 'L2 or L1 regularizer')


tf.compat.v1.app.flags.DEFINE_string('model_name', 'snn_train_mlp_mnist', 'model name')

tf.compat.v1.app.flags.DEFINE_string('n_type', 'LIF', 'LIF or IF: neuron type')

#
tf.compat.v1.app.flags.DEFINE_string('dataset', 'MNIST', 'dataset')
tf.compat.v1.app.flags.DEFINE_string('ann_model', 'MLP', 'neural network model')

#
tf.compat.v1.app.flags.DEFINE_boolean('verbose',False, 'verbose mode')
tf.compat.v1.app.flags.DEFINE_boolean('verbose_visual',False, 'verbose visual mode')

#
tf.compat.v1.app.flags.DEFINE_integer('time_step_save_interval',10,'snn test save interval')

#
tf.compat.v1.app.flags.DEFINE_bool('f_fused_bn',False,'f_fused_bn')

# data-based normalization
tf.compat.v1.app.flags.DEFINE_bool('f_stat_train_mode',False,'f_stat_train_mode')
tf.compat.v1.app.flags.DEFINE_bool('f_vth_conp',False,'f_vth_conp')
tf.compat.v1.app.flags.DEFINE_bool('f_spike_max_pool',False,'f_spike_max_pool')
tf.compat.v1.app.flags.DEFINE_bool('f_w_norm_data',False,'f_w_norm_data')
tf.compat.v1.app.flags.DEFINE_float('p_ws',8,'period of wieghted synapse')

tf.compat.v1.app.flags.DEFINE_integer('num_class',10,'number_of_class (do not touch)')

#
tf.compat.v1.app.flags.DEFINE_string('input_spike_mode','POISSON','input spike mode - POISSON, WEIGHTED_SPIKE, ROPOSED')
tf.compat.v1.app.flags.DEFINE_string('neural_coding','RATE','neural coding - RATE, WEIGHTED_SPIKE, PROPOSED')

tf.compat.v1.app.flags.DEFINE_bool('f_positive_vmem',False,'positive vmem')
tf.compat.v1.app.flags.DEFINE_bool('f_tot_psp',False,'accumulate total psp')

tf.compat.v1.app.flags.DEFINE_bool('f_refractory',False,'refractory mode')

tf.compat.v1.app.flags.DEFINE_bool('f_write_stat',False,'write stat')
tf.compat.v1.app.flags.DEFINE_string('act_save_mode','channel','activation save mode')
tf.compat.v1.app.flags.DEFINE_bool('f_save_result',True,'save result to xlsx file')


#tf.compat.v1.app.flags.DEFINE_integer('k_pathces', 5, 'patches for test (random crop)')
tf.compat.v1.app.flags.DEFINE_integer('input_size', 28, 'input image width / height')


#
tf.compat.v1.app.flags.DEFINE_string('path_stat','./stat/', 'path stat')
tf.compat.v1.app.flags.DEFINE_string('prefix_stat','act_n_train', 'prefix of stat file name')


#
tf.compat.v1.app.flags.DEFINE_bool('f_data_std', True, 'data_standardization')


tf.compat.v1.app.flags.DEFINE_string('path_result_root','./result/', 'path result root')

# temporal coding
tf.compat.v1.app.flags.DEFINE_integer('tc',10,'time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_integer('time_window',20,'time window of each layer for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_duration',20,'time fire duration for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_visual_record_first_spike_time',False,'flag - visual recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_train_time_const',False,'flag - enable to train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_train_time_const_outlier',True,'flag - enable to outlier roubst train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_load_time_const',False,'flag - load time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_string('time_const_init_file_name','./temporal_coding/time_const','temporal coding file name - time_const, time_delay`')
tf.compat.v1.app.flags.DEFINE_integer('time_const_num_trained_data',0,'number of trained data - time constant')
tf.compat.v1.app.flags.DEFINE_integer('time_const_save_interval',10000,'save interval - training time constant')
tf.compat.v1.app.flags.DEFINE_integer('epoch_train_time_const',1,'epoch - training time constant')



#
tf.compat.v1.app.flags.DEFINE_enum('snn_output_type',"VMEM", ["SPIKE", "VMEM", "FIRST_SPIKE_TIME"], "snn output type")


#
tf.compat.v1.app.flags.DEFINE_bool("en_tensorboard_write", False, "Tensorboard write")


################################################################################
# Deep SNNs training w/ tepmoral information - surrogate DNN model
################################################################################

#
conf = flags.FLAGS


#
#conf.time_fire_start = 1.5

# TODO: parameterize - input 0~1
if conf.model_name == 'vgg_cifar_ro_0':
    conf.f_data_std = False


# stat mode - cpu
if conf.f_write_stat:
    en_gpu=False

def main(_):
    print('main start')

    # remove output dir
    if conf.en_remove_output_dir:
        print('remove output dir: %s' % conf.output_dir)
        shutil.rmtree(conf.output_dir,ignore_errors=True)

    data_format = 'channels_last'

    if en_gpu==True:
        (device, data_format) = ('/gpu:0', data_format)
    else:
        (device, data_format) = ('/cpu:0', data_format)

    print ('Using device %s, and data format %s.' %(device, data_format))

    with tf.device('/gpu:0'):
        dataset_type= {
            'MNIST': mnist,
            'CIFAR-10': cifar10,
            'CIFAR-100': cifar100,
        }
        dataset = dataset_type[conf.dataset]

        (train_dataset, val_dataset, test_dataset, num_train_dataset, num_val_dataset, num_test_dataset) = dataset.load(conf)

    model = None
    if conf.ann_model=='CNN':
        if conf.dataset=='MNIST':
            #model = model_cnn_mnist.MNISTModel_CNN(data_format,conf)
            model = cnn_mnist.MNISTModel_CNN(data_format,conf)
    elif conf.ann_model=='VGG16':
        if conf.dataset=='CIFAR-10':
            model = VGG16.CIFARModel_CNN(data_format,conf)
        elif conf.dataset=='CIFAR-100':
            model = VGG16.CIFARModel_CNN(data_format,conf)

    if model is None:
        print('not supported model name: '+conf.ann_model)
        os._exit(0)


    #
    save_target_acc_sel = {
        'MNIST': 90.0,
        'CIFAR-10': 91.0,
        'CIFAR-100': 68.0
    }
    save_target_acc = save_target_acc_sel[conf.dataset]

    en_train = conf.en_train

    #
    if en_train:
        if(conf.nn_mode=="ANN"):
            train_func = train.train_one_epoch(conf.nn_mode)
        else:
            print("error in nn_mode: %s"%(conf.nn_mode))
            assert(False)


    #
    lr=tf.Variable(conf.lr)
    optimizer = tf.keras.optimizers.Adam(lr)


    if conf.output_dir:
        output_dir = os.path.join(conf.output_dir,conf.model_name+'_'+conf.nn_mode)
        output_dir = os.path.join(output_dir,now.strftime("%Y%m%d-%H%M"))

        train_dir = os.path.join(output_dir,'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')

        if not os.path.isdir(output_dir):
            tf.io.gfile.makedirs(output_dir)
    else:
        train_dir = None
        val_dir = None
        val_snn_dir = None
        test_dir = None

    #
    summary_writer = tf.summary.create_file_writer(train_dir,flush_millis=100)
    val_summary_writer = tf.summary.create_file_writer(val_dir,flush_millis=100,name='val')


    # TODO: TF-V1
    #test_summary_writer = tf.contrib.summary.create_file_writer(test_dir,flush_millis=100,name='test')
    test_summary_writer = tf.summary.create_file_writer(test_dir,flush_millis=100,name='test')


    checkpoint_dir = os.path.join(conf.checkpoint_dir,conf.model_name)
    checkpoint_load_dir = os.path.join(conf.checkpoint_load_dir,conf.model_name)

    print('model load path: %s' % checkpoint_load_dir)
    print('model save path: %s' % checkpoint_dir)

    if en_train:
        # force to overwrite train model
        if not conf.en_load_model:
            print('remove pre-trained model: {}'.format(checkpoint_dir))
            shutil.rmtree(checkpoint_dir,ignore_errors=True)

        if not os.path.isdir(checkpoint_dir):
            tf.io.gfile.makedirs(checkpoint_dir)

    if not os.path.isdir(checkpoint_load_dir):
        print('there is no load dir: %s' % checkpoint_load_dir)
        sys.exit(1)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')


    # epoch
    global_epoch = tf.Variable(name='global_epoch', initial_value=tf.zeros(shape=[]),dtype=tf.float32,trainable=False)

    with tf.device(device):
        if en_train:
            print('Train Phase >')

            acc_val_target_best = 0.0
            acc_val_best = 0.0
            acc_val_snn_best = 0.0

            if conf.dataset!='ImageNet':
                train_dataset_p = dataset.train_data_augmentation(train_dataset, conf.batch_size)

            images_0 = next(train_dataset_p.__iter__())[0]


            model(images_0,False)

            if conf.en_load_model:
                restore_variables = (model.trainable_weights + optimizer.variables() + [global_epoch])

                print('load model')
                print(tf.train.latest_checkpoint(checkpoint_load_dir))

                load_layer = model.load_layer_ann_checkpoint
                load_model = tf.train.Checkpoint(model=load_layer, optimizer=optimizer, global_epoch=global_epoch)
                load_model.restore(tf.train.latest_checkpoint(checkpoint_dir))

                print('load model done')
                epoch_start=int(global_epoch.numpy())

            else:
                epoch_start = 0

            #
            for epoch in range(epoch_start,epoch_start+conf.epoch+1):
                with summary_writer.as_default():
                    loss_train, acc_train = train_func(model, optimizer, train_dataset_p)
                    save_epoch=1

                    if conf.en_tensorboard_write:
                        tf.summary.scalar('loss', loss_train, step=epoch)
                        tf.summary.scalar('accuracy', acc_train, step=epoch)

                #
                f_save_model = False
                with val_summary_writer.as_default():
                    loss_val, acc_val, _ = test.test(model, val_dataset, num_val_dataset, conf, f_val=True, epoch=epoch)

                    if conf.en_tensorboard_write:
                        tf.summary.scalar('loss', loss_val, step=epoch)
                        tf.summary.scalar('accuracy', acc_val, step=epoch)

                    if acc_val_best < acc_val:
                        acc_val_best = acc_val
                        f_save_model = True

                    acc_val_target_best = acc_val_best

                    #
                    if f_save_model:
                        if epoch > epoch_start+save_epoch:
                            f_save_model = acc_val_target_best > save_target_acc
                            if f_save_model:
                                print('save model')
                                global_epoch.assign(epoch)

                                # save model
                                checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, global_epoch=global_epoch)
                                checkpoint.save(file_prefix=checkpoint_prefix)

                print('[%3d] train(loss: %.3f, acc: %.3f), valid(loss: %.3f, acc: %.3f, best: %.3f)'%(epoch,loss_train,acc_train,loss_val,acc_val,acc_val_best))


        #
        print(' Test Phase >')

        if en_train == False:
            if conf.dataset=='ImageNet':
                images_0 = next(test_dataset.__iter__())[0]
            else:
                images_0 = next(test_dataset.__iter__())[0]
            # dummy run
            model(images_0,False)

            load_layer = model.load_layer_ann_checkpoint
            load_model = tf.train.Checkpoint(model=load_layer, optimizer=optimizer, global_epoch=global_epoch)

            status = load_model.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
            print('load model done')

            if conf.f_train_time_const:
                for epoch in range(conf.epoch_train_time_const):
                    print("epoch: {:d}".format(epoch))
                    with test_summary_writer.as_default():
                        loss_test, acc_test, acc_test_top5 = test.test(model, test_dataset,num_test_dataset, conf, epoch=epoch)
                        if conf.dataset == 'ImageNet':
                            print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
                        else:
                            print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))
            else:
                #
                with test_summary_writer.as_default():
                    loss_test, acc_test, acc_test_top5 = test.test(model, test_dataset, num_test_dataset, conf)
                    #loss_test, acc_test, acc_test_top5 = test.test(model, val_dataset, num_val_dataset, conf)
                    if conf.dataset == 'ImageNet':
                        print('loss_test: %f, acc_test: %f, acc_test_top5: %f'%(loss_test,acc_test,acc_test_top5))
                    else:
                        print('loss_test: %f, acc_test: %f'%(loss_test,acc_test))

        print('end')


if __name__ == '__main__':
    tf.compat.v1.app.run()

