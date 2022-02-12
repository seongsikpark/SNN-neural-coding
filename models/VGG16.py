import tensorflow as tf
#import tensorflow.contrib.eager as tfe

#from tensorflow.contrib.layers.python.layers import initializers
#from tensorflow.contrib.layers.python.layers import regularizers

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers


from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops

import tensorflow_probability as tfp
tfd = tfp.distributions

#
import util
import lib_snn
import sys
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import math
import csv
import collections

from scipy import stats
from scipy import sparse

from operator import itemgetter

from functools import partial

#import tfplot
import threading

#
# noinspection PyUnboundLocalVariable
#class CIFARModel_CNN(tfe.Network):
#class CIFARModel_CNN(tf.keras.layers):
class CIFARModel_CNN(tf.keras.layers.Layer):
    def __init__(self, data_format, conf):
        super(CIFARModel_CNN, self).__init__(name='')

        self.data_format = data_format
        self.conf = conf
        self.num_class = self.conf.num_class

        self.f_1st_iter = True
        self.f_load_model_done = False
        self.verbose = conf.verbose
        self.f_debug_visual = conf.verbose_visual
        self.f_done_preproc = False

        self.kernel_size = 3
        self.fanin_conv = self.kernel_size*self.kernel_size
        #self.fanin_conv = self.kernel_size*self.kernel_size/9

        self.tw=conf.time_step

        self.count_accuracy_time_point=0
        self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        self.accuracy_time_point.append(conf.time_step)
        #self.num_accuracy_time_point = int(math.ceil(float(conf.time_step)/float(conf.time_step_save_interval))
        self.num_accuracy_time_point = len(self.accuracy_time_point)


        if self.f_debug_visual:
            #self.debug_visual_threads = []
            self.debug_visual_axes = []
            self.debug_visual_list_neuron = collections.OrderedDict()

        #
        self.f_skip_bn = self.conf.f_fused_bn


        #
        self.epoch = -1

        #
        self.layer_name=[
            #'in',
            'conv1',
            'conv1_1',
            'conv2',
            'conv2_1',
            'conv3',
            'conv3_1',
            'conv3_2',
            'conv4',
            'conv4_1',
            'conv4_2',
            'conv5',
            'conv5_1',
            'conv5_2',
            'fc1',
            'fc2',
            'fc3',
        ]

        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])

        self.total_residual_vmem=np.zeros(len(self.layer_name)+1)

        self.output_layer_last_spike_time=np.zeros(self.num_class)

        # nomarlization factor
        self.norm=collections.OrderedDict()
        self.norm_b=collections.OrderedDict()

        #
        if self.data_format == 'channels_first':
            self._input_shape = [-1,3,32,32]    # CIFAR-10
            #self._input_shape = [-1,3,cifar_10_crop_size,cifar_10_crop_size]    # CIFAR-10
        else:
            assert self.data_format == 'channels_last'
            self._input_shape = [-1,32,32,3]
            #self._input_shape = [-1,cifar_10_crop_size,cifar_10_crop_size,3]


        if conf.nn_mode == 'ANN':
            use_bias = conf.use_bias
        else :
            #use_bias = False
            use_bias = conf.use_bias

        #activation = tf.nn.relu
        activation = None
        padding = 'same'

        self.run_mode = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        self.run_mode_load_model = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        regularizer_type = {
            'L1': regularizers.l1(conf.lamb),
            'L2': regularizers.l2(conf.lamb)
        }


        kernel_regularizer = regularizer_type[self.conf.regularizer]
        #kernel_initializer = initializers.xavier_initializer(True)
        kernel_initializer = initializers.GlorotUniform()
        #kernel_initializer = initializers.variance_scaling_initializer(factor=2.0,mode='FAN_IN')    # MSRA init. = He init

        self.list_layer=collections.OrderedDict()
        self.list_layer['conv1'] = tf.keras.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv1_1'] = tf.keras.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv1_1_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv2'] = tf.keras.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv2_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv2_1'] = tf.keras.layers.Conv2D(128,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv2_1_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv3'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv3_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv3_1'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv3_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv3_2'] = tf.keras.layers.Conv2D(256,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv3_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv4'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv4_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv4_1'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv4_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv4_2'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv4_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['conv5'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv5_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv5_1'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv5_1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv5_2'] = tf.keras.layers.Conv2D(512,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='SAME')
        self.list_layer['conv5_2_bn'] = tf.keras.layers.BatchNormalization()

        self.list_layer['fc1'] = tf.keras.layers.Dense(512,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)

        self.list_layer['fc1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['fc2'] = tf.keras.layers.Dense(512,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.list_layer['fc2_bn'] = tf.keras.layers.BatchNormalization()

        #self.list_layer['fc3'] = tf.keras.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        self.list_layer['fc3'] = tf.keras.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.list_layer['fc3_bn'] = tf.keras.layers.BatchNormalization()


        if self.conf.dataset=='CIFAR-100':
            self.dropout_conv = tf.keras.layers.Dropout(0.3)
            self.dropout_conv2 = tf.keras.layers.Dropout(0.3)
            self.dropout = tf.keras.layers.Dropout(0.3)
        else:
            # cifar-10
            self.dropout_conv = tf.keras.layers.Dropout(0.3)
            self.dropout_conv2 = tf.keras.layers.Dropout(0.4)
            self.dropout = tf.keras.layers.Dropout(0.5)


        # remove later
        self.conv1=self.list_layer['conv1']
        self.conv1_bn=self.list_layer['conv1_bn']
        self.conv1_1=self.list_layer['conv1_1']
        self.conv1_1_bn=self.list_layer['conv1_1_bn']
        self.conv2=self.list_layer['conv2']
        self.conv2_bn=self.list_layer['conv2_bn']
        self.conv2_1=self.list_layer['conv2_1']
        self.conv2_1_bn=self.list_layer['conv2_1_bn']
        self.conv3=self.list_layer['conv3']
        self.conv3_bn=self.list_layer['conv3_bn']
        self.conv3_1=self.list_layer['conv3_1']
        self.conv3_1_bn=self.list_layer['conv3_1_bn']
        self.conv3_2=self.list_layer['conv3_2']
        self.conv3_2_bn=self.list_layer['conv3_2_bn']
        self.conv4=self.list_layer['conv4']
        self.conv4_bn=self.list_layer['conv4_bn']
        self.conv4_1=self.list_layer['conv4_1']
        self.conv4_1_bn=self.list_layer['conv4_1_bn']
        self.conv4_2=self.list_layer['conv4_2']
        self.conv4_2_bn=self.list_layer['conv4_2_bn']
        self.conv5=self.list_layer['conv5']
        self.conv5_bn=self.list_layer['conv5_bn']
        self.conv5_1=self.list_layer['conv5_1']
        self.conv5_1_bn=self.list_layer['conv5_1_bn']
        self.conv5_2=self.list_layer['conv5_2']
        self.conv5_2_bn=self.list_layer['conv5_2_bn']
        self.fc1=self.list_layer['fc1']
        self.fc1_bn=self.list_layer['fc1_bn']
        self.fc2=self.list_layer['fc2']
        self.fc2_bn=self.list_layer['fc2_bn']
        self.fc3=self.list_layer['fc3']
        self.fc3_bn=self.list_layer['fc3_bn']

        self.pool2d = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME', data_format=data_format)
        self.act_relu = tf.nn.relu


        self.in_shape = [self.conf.batch_size]+self._input_shape[1:]

        #self.shape_out_conv1 = util.cal_output_shape_Conv2D(self.data_format,input_shape_one_sample,64,3,1)
        self.shape_out_conv1 = util.cal_output_shape_Conv2D(self.data_format,self.in_shape,64,self.kernel_size,1)
        self.shape_out_conv1_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv1,64,self.kernel_size,1)
        self.shape_out_conv1_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv1_1,2,2)

        self.shape_out_conv2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv1_p,128,self.kernel_size,1)
        self.shape_out_conv2_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv2,128,self.kernel_size,1)
        self.shape_out_conv2_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv2_1,2,2)

        self.shape_out_conv3 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv2_p,256,self.kernel_size,1)
        self.shape_out_conv3_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv3,256,self.kernel_size,1)
        self.shape_out_conv3_2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv3_1,256,self.kernel_size,1)
        self.shape_out_conv3_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv3_2,2,2)

        self.shape_out_conv4 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv3_p,512,self.kernel_size,1)
        self.shape_out_conv4_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv4,512,self.kernel_size,1)
        self.shape_out_conv4_2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv4_1,512,self.kernel_size,1)
        self.shape_out_conv4_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv4_2,2,2)

        self.shape_out_conv5 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv4_p,512,self.kernel_size,1)
        self.shape_out_conv5_1 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv5,512,self.kernel_size,1)
        self.shape_out_conv5_2 = util.cal_output_shape_Conv2D(self.data_format,self.shape_out_conv5_1,512,self.kernel_size,1)
        self.shape_out_conv5_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv5_2,2,2)

        self.shape_out_fc1 = tensor_shape.TensorShape([self.conf.batch_size,512]).as_list()
        self.shape_out_fc2 = tensor_shape.TensorShape([self.conf.batch_size,512]).as_list()
        self.shape_out_fc3 = tensor_shape.TensorShape([self.conf.batch_size,self.num_class]).as_list()


        self.dict_shape=collections.OrderedDict()
        self.dict_shape['conv1']=self.shape_out_conv1
        self.dict_shape['conv1_1']=self.shape_out_conv1_1
        self.dict_shape['conv1_p']=self.shape_out_conv1_p
        self.dict_shape['conv2']=self.shape_out_conv2
        self.dict_shape['conv2_1']=self.shape_out_conv2_1
        self.dict_shape['conv2_p']=self.shape_out_conv2_p
        self.dict_shape['conv3']=self.shape_out_conv3
        self.dict_shape['conv3_1']=self.shape_out_conv3_1
        self.dict_shape['conv3_2']=self.shape_out_conv3_2
        self.dict_shape['conv3_p']=self.shape_out_conv3_p
        self.dict_shape['conv4']=self.shape_out_conv4
        self.dict_shape['conv4_1']=self.shape_out_conv4_1
        self.dict_shape['conv4_2']=self.shape_out_conv4_2
        self.dict_shape['conv4_p']=self.shape_out_conv4_p
        self.dict_shape['conv5']=self.shape_out_conv5
        self.dict_shape['conv5_1']=self.shape_out_conv5_1
        self.dict_shape['conv5_2']=self.shape_out_conv5_2
        self.dict_shape['conv5_p']=self.shape_out_conv5_p
        self.dict_shape['fc1']=self.shape_out_fc1
        self.dict_shape['fc2']=self.shape_out_fc2
        self.dict_shape['fc3']=self.shape_out_fc3


        self.dict_shape_one_batch=collections.OrderedDict()
        for l_name in self.layer_name:
            self.dict_shape_one_batch[l_name] = tensor_shape.TensorShape([1,]+self.dict_shape[l_name][1:])


        #
        self.dict_stat_r=collections.OrderedDict()  # read
        self.dict_stat_w=collections.OrderedDict()  # write


        #
        if self.conf.f_write_stat:

            #
            self.layer_name_write_stat=[
                #'in',
                'conv1',
                'conv1_1',
                'conv2',
                'conv2_1',
                'conv3',
                'conv3_1',
                'conv3_2',
                'conv4',
                'conv4_1',
                'conv4_2',
                'conv5',
                'conv5_1',
                'conv5_2',
                'fc1',
                'fc2',
                'fc3',
            ]

            self.f_1st_iter_stat = True

            for l_name in self.layer_name_write_stat:
                self.dict_stat_w[l_name]=tf.Variable(initial_value=tf.zeros(self.dict_shape[l_name]),trainable=None)


        self.conv_p=collections.OrderedDict()
        self.conv_p['conv1_p']=np.empty(self.dict_shape['conv1_p'],dtype=np.float32)
        self.conv_p['conv2_p']=np.empty(self.dict_shape['conv2_p'],dtype=np.float32)
        self.conv_p['conv3_p']=np.empty(self.dict_shape['conv3_p'],dtype=np.float32)
        self.conv_p['conv4_p']=np.empty(self.dict_shape['conv4_p'],dtype=np.float32)
        self.conv_p['conv5_p']=np.empty(self.dict_shape['conv5_p'],dtype=np.float32)


        #
        if (self.conf.nn_mode=='SNN' and self.conf.f_train_time_const) or (self.conf.nn_mode=='ANN' and self.conf.f_write_stat):
            self.dnn_act_list=collections.OrderedDict()

        # neurons
        if self.conf.nn_mode == 'SNN':
            print('Neuron setup')

            self.input_shape_snn = [self.conf.batch_size] + self._input_shape[1:]

            print('Input shape snn: '+str(self.input_shape_snn))

            n_type = self.conf.n_type
            nc = self.conf.neural_coding

            self.list_neuron=collections.OrderedDict()

            self.list_neuron['in'] = lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf,nc,0,'in')

            self.list_neuron['conv1'] = lib_snn.Neuron(self.shape_out_conv1,n_type,self.fanin_conv,self.conf,nc,1,'conv1')
            self.list_neuron['conv1_1'] = lib_snn.Neuron(self.shape_out_conv1_1,n_type,self.fanin_conv,self.conf,nc,2,'conv1_1')

            self.list_neuron['conv2'] = lib_snn.Neuron(self.shape_out_conv2,n_type,self.fanin_conv,self.conf,nc,3,'conv2')
            self.list_neuron['conv2_1'] = lib_snn.Neuron(self.shape_out_conv2_1,n_type,self.fanin_conv,self.conf,nc,4,'conv2_1')

            self.list_neuron['conv3'] = lib_snn.Neuron(self.shape_out_conv3,n_type,self.fanin_conv,self.conf,nc,5,'conv3')
            self.list_neuron['conv3_1'] = lib_snn.Neuron(self.shape_out_conv3_1,n_type,self.fanin_conv,self.conf,nc,6,'conv3_1')
            self.list_neuron['conv3_2'] = lib_snn.Neuron(self.shape_out_conv3_2,n_type,self.fanin_conv,self.conf,nc,7,'conv3_2')

            self.list_neuron['conv4'] = lib_snn.Neuron(self.shape_out_conv4,n_type,self.fanin_conv,self.conf,nc,8,'conv4')
            self.list_neuron['conv4_1'] = lib_snn.Neuron(self.shape_out_conv4_1,n_type,self.fanin_conv,self.conf,nc,9,'conv4_1')
            self.list_neuron['conv4_2'] = lib_snn.Neuron(self.shape_out_conv4_2,n_type,self.fanin_conv,self.conf,nc,10,'conv4_2')

            self.list_neuron['conv5'] = lib_snn.Neuron(self.shape_out_conv5,n_type,self.fanin_conv,self.conf,nc,11,'conv5')
            self.list_neuron['conv5_1'] = lib_snn.Neuron(self.shape_out_conv5_1,n_type,self.fanin_conv,self.conf,nc,12,'conv5_1')
            self.list_neuron['conv5_2'] = lib_snn.Neuron(self.shape_out_conv5_2,n_type,self.fanin_conv,self.conf,nc,13,'conv5_2')

            self.list_neuron['fc1'] = lib_snn.Neuron(self.shape_out_fc1,n_type,512,self.conf,nc,14,'fc1')
            self.list_neuron['fc2'] = lib_snn.Neuron(self.shape_out_fc2,n_type,512,self.conf,nc,15,'fc2')
            self.list_neuron['fc3'] = lib_snn.Neuron(self.shape_out_fc3,'OUT',512,self.conf,nc,16,'fc3')


            # modify later
            self.n_in = self.list_neuron['in']

            self.n_conv1 = self.list_neuron['conv1']
            self.n_conv1_1 = self.list_neuron['conv1_1']

            self.n_conv2 = self.list_neuron['conv2']
            self.n_conv2_1 = self.list_neuron['conv2_1']

            self.n_conv3 = self.list_neuron['conv3']
            self.n_conv3_1 = self.list_neuron['conv3_1']
            self.n_conv3_2 = self.list_neuron['conv3_2']

            self.n_conv4 = self.list_neuron['conv4']
            self.n_conv4_1 = self.list_neuron['conv4_1']
            self.n_conv4_2 = self.list_neuron['conv4_2']

            self.n_conv5 = self.list_neuron['conv5']
            self.n_conv5_1 = self.list_neuron['conv5_1']
            self.n_conv5_2 = self.list_neuron['conv5_2']

            self.n_fc1 = self.list_neuron['fc1']
            self.n_fc2 = self.list_neuron['fc2']
            self.n_fc3 = self.list_neuron['fc3']

            #
            self.snn_output_layer = self.n_fc3

            self.snn_output = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_layer.dim)),dtype=tf.float32,trainable=False)


            if self.conf.neural_coding=='TEMPORAL' and self.conf.f_load_time_const:

                file_name = self.conf.time_const_init_file_name
                file_name = file_name + '/'+self.conf.model_name
                #if self.conf.f_tc_based:
                if False:
                    file_name = file_name+'/tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.n_tau_time_window)+'_tau_itr-'+str(self.conf.time_const_num_trained_data)
                else:
                    file_name = file_name+'/tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.time_window)+'_itr-'+str(self.conf.time_const_num_trained_data)

                if conf.f_train_time_const_outlier:
                    file_name+="_outlier"

                print('load trained time constant: file_name: {:s}'.format(file_name))

                file = open(file_name,'r')
                lines = csv.reader(file)

                for line in lines:
                    if not line:
                        continue

                    print(line)

                    type = line[0]
                    name = line[1]
                    val = float(line[2])

                    if (type=='tc') :

                        self.list_neuron[name].set_time_const_init_fire(val)

                        if not ('in' in name):
                            self.list_neuron[name].set_time_const_init_integ(self.list_neuron[name_prev].time_const_init_fire)

                        name_prev = name

                    elif (type=='td'):

                        self.list_neuron[name].set_time_delay_init_fire(val)

                        if not ('in' in name):
                            self.list_neuron[name].set_time_delay_init_integ(self.list_neuron[name_prev].time_delay_init_fire)

                        name_prev = name

                    else:
                        print("not supported temporal coding type")
                        assert(False)


                file.close()


            #
            self.spike_count = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_fc3.dim)),dtype=tf.float32,trainable=False)
            #self.spike_count = tf.contrib.eager.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_fc3.dim)),dtype=tf.float32,trainable=False)
        #
        self.cmap=matplotlib.cm.get_cmap('viridis')
        #self.normalize=matplotlib.colors.Normalize(vmin=min(self.n_fc3.vmem),vmax=max(self.n_fc3.vmem))

        # model loading V2
        self.load_layer_ann_checkpoint = self.load_layer_ann_checkpoint_func()

    #
    def dist_beta_sample_func(self):
        if self.conf.f_loss_enc_spike:
            for l_name, tk in self.list_tk.items():
                enc_st = tf.reshape(tk.out_enc, [-1])

                samples = self.dist.sample(enc_st.shape)
                samples = tf.multiply(samples,self.enc_st_target_end)
                self.dist_beta_sample[l_name] = tf.histogram_fixed_width(samples, [0,self.enc_st_target_end], nbins=self.enc_st_target_end)
        else:
            pass

    #
    def load_layer_ann_checkpoint_func(self):
        load_layer_ann_checkpoint = tf.train.Checkpoint(
            conv1=self.conv1,
            conv1_bn=self.conv1_bn,
            conv1_1=self.conv1_1,
            conv1_1_bn=self.conv1_1_bn,
            conv2=self.conv2,
            conv2_bn=self.conv2_bn,
            conv2_1=self.conv2_1,
            conv2_1_bn=self.conv2_1_bn,
            conv3=self.conv3,
            conv3_bn=self.conv3_bn,
            conv3_1=self.conv3_1,
            conv3_1_bn=self.conv3_1_bn,
            conv3_2=self.conv3_2,
            conv3_2_bn=self.conv3_2_bn,
            conv4=self.conv4,
            conv4_bn=self.conv4_bn,
            conv4_1=self.conv4_1,
            conv4_1_bn=self.conv4_1_bn,
            conv4_2=self.conv4_2,
            conv4_2_bn=self.conv4_2_bn,
            conv5=self.conv5,
            conv5_bn=self.conv5_bn,
            conv5_1=self.conv5_1,
            conv5_1_bn=self.conv5_1_bn,
            conv5_2=self.conv5_2,
            conv5_2_bn=self.conv5_2_bn,
            fc1=self.fc1,
            fc1_bn=self.fc1_bn,
            fc2=self.fc2,
            fc2_bn=self.fc2_bn,
            fc3=self.fc3,
            fc3_bn=self.fc3_bn
        )

        return load_layer_ann_checkpoint


    ###########################################################################
    ## processing
    ###########################################################################

    def reset_per_run_snn(self):
        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])


    def reset_per_sample_snn(self):
        self.reset_neuron()
        #self.snn_output = np.zeros((self.num_accuracy_time_point,)+self.n_fc1.get_spike_count().numpy().shape)
        self.snn_output.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_layer.dim)))
        self.count_accuracy_time_point=0


    def reset_neuron(self):
        self.n_in.reset()
        self.n_conv1.reset()
        self.n_conv2.reset()
        self.n_fc1.reset()


    def preproc(self, inputs, f_training, f_val_snn=False):
        preproc_sel= {
            'ANN': self.preproc_ann,
            'SNN': self.preproc_snn
        }

        if f_val_snn:
            self.preproc_snn(inputs,f_training)
        else:
            preproc_sel[self.conf.nn_mode](inputs, f_training)



    def preproc_snn(self,inputs,f_training):
        # reset for sample
        self.reset_per_sample_snn()

        if self.f_done_preproc == False:
            self.f_done_preproc = True
            #self.print_model_conf()
            self.reset_per_run_snn()
            self.preproc_ann_to_snn()

        # gradient-based optimization of TC and td in temporal coding (TTFS)
        if (self.conf.neural_coding=="TEMPORAL" and self.conf.f_train_time_const):
            self.call_ann(inputs,f_training)

    def preproc_ann(self, inputs, f_training):
        if self.f_done_preproc == False:
            self.f_done_preproc=True
            self.print_model_conf()
            self.preproc_ann_norm()

        self.f_skip_bn=self.conf.f_fused_bn


    def preproc_ann_to_snn(self):
        if self.conf.verbose:
            print('preprocessing: ANN to SNN')

        if self.conf.f_fused_bn or ((self.conf.nn_mode=='ANN')and(self.conf.f_validation_snn)):
            self.fused_bn()


        #print(np.max(self.list_layer.values()[0].kernel))
        #self.temporal_norm()
        #print(np.max(self.list_layer.values()[0].kernel))


        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()


        #self.print_act_after_w_norm()

    def preproc_surrogate_training_model(self):
        self.dist_beta_sample_func()



    #
    # TODO: input neuron ?
    def load_temporal_kernel_para(self):
        if self.conf.verbose:
            print('preprocessing: load_temporal_kernel_para')

        for l_name in self.layer_name:

            if l_name != self.layer_name[-1]:
                self.list_neuron[l_name].set_time_const_fire(self.list_tk[l_name].tc)
                self.list_neuron[l_name].set_time_delay_fire(self.list_tk[l_name].td)

            if not ('in' in l_name):
                #self.list_neuron[l_name].set_time_const_integ(self.list_tk[l_name_prev].tc_dec)
                #self.list_neuron[l_name].set_time_delay_integ(self.list_tk[l_name_prev].td_dec)
                self.list_neuron[l_name].set_time_const_integ(self.list_tk[l_name_prev].tc)
                self.list_neuron[l_name].set_time_delay_integ(self.list_tk[l_name_prev].td)

            l_name_prev = l_name

        # encoding decoding kernerl seperate
        #assert(False)




    #
    def preproc_ann_norm(self):
        if self.conf.f_fused_bn:
            self.fused_bn()

        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()

    #
    def call(self, inputs, f_training, epoch=-1, f_val_snn=False):

        if self.f_load_model_done:

            # pre-processing
            self.preproc(inputs,f_training,f_val_snn)

            # run
            if (self.conf.nn_mode=='SNN' and self.conf.neural_coding=="TEMPORAL" and self.conf.f_train_time_const):
                # inference - temporal coding
                #ret_val = self.call_snn_temporal(inputs,f_training,self.conf.time_step)

                #
                if self.conf.f_train_time_const:
                    self.run_mode['ANN'](inputs,f_training,self.conf.time_step,epoch)

                ret_val = self.run_mode[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)

                # training time constant
                if self.conf.f_train_time_const:
                    self.train_time_const()
            else:
                # inference - rate, phase burst coding
                ret_val = self.run_mode[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)


            # post-processing
            #self.postproc(f_val_snn)
        else:

            ret_val = self.run_mode_load_model[self.conf.nn_mode](inputs,f_training,self.conf.time_step,epoch)
            self.f_load_model_done=True
        return ret_val


    #
    def fused_bn(self):
        print('fused_bn')
        self.conv_bn_fused(self.conv1, self.conv1_bn, 1.0)
        self.conv_bn_fused(self.conv1_1, self.conv1_1_bn, 1.0)
        self.conv_bn_fused(self.conv2, self.conv2_bn, 1.0)
        self.conv_bn_fused(self.conv2_1, self.conv2_1_bn, 1.0)
        self.conv_bn_fused(self.conv3, self.conv3_bn, 1.0)
        self.conv_bn_fused(self.conv3_1, self.conv3_1_bn, 1.0)
        self.conv_bn_fused(self.conv3_2, self.conv3_2_bn, 1.0)
        self.conv_bn_fused(self.conv4, self.conv4_bn, 1.0)
        self.conv_bn_fused(self.conv4_1, self.conv4_1_bn, 1.0)
        self.conv_bn_fused(self.conv4_2, self.conv4_2_bn, 1.0)
        self.conv_bn_fused(self.conv5, self.conv5_bn, 1.0)
        self.conv_bn_fused(self.conv5_1, self.conv5_1_bn, 1.0)
        self.conv_bn_fused(self.conv5_2, self.conv5_2_bn, 1.0)
        self.fc_bn_fused(self.fc1, self.fc1_bn, 1.0)
        self.fc_bn_fused(self.fc2, self.fc2_bn, 1.0)
        #if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name):
        self.fc_bn_fused(self.fc3, self.fc3_bn, 1.0)

    #
    def w_norm_layer_wise(self):
        f_norm=np.max

        for idx_l, l in enumerate(self.layer_name):
            if idx_l==0:
                self.norm[l]=f_norm(self.dict_stat_r[l])
            else:
                self.norm[l]=f_norm(list(self.dict_stat_r.values())[idx_l])/f_norm(list(self.dict_stat_r.values())[idx_l-1])

            self.norm_b[l]=f_norm(self.dict_stat_r[l])

        if self.conf.f_vth_conp:
            for idx_l, l in enumerate(self.layer_name):
                #self.list_neuron[l].set_vth(np.broadcast_to(self.conf.n_init_vth*1.0 + 0.1*self.dict_stat_r[l]/f_norm(self.dict_stat_r[l]),self.dict_shape[l]))
                self.list_neuron[l].set_vth(np.broadcast_to(self.dict_stat_r[l]/f_norm(self.dict_stat_r[l]),self.dict_shape[l]))
                #self.list_neuron[l].set_vth(np.broadcast_to(self.dict_stat_r[l]/np.broadcast_to(f_norm(self.dict_stat_r[l]),self.dict_stat_r[l].shape)   ,self.dict_shape[l]))

        #self.print_act_d()
        # print
        for k, v in self.norm.items():
            print(k +': '+str(v))

        for k, v in self.norm_b.items():
            print(k +': '+str(v))

        #
        self.conv1.kernel = self.conv1.kernel/self.norm['conv1']
        self.conv1.bias = self.conv1.bias/self.norm_b['conv1']
        self.conv1_1.kernel = self.conv1_1.kernel/self.norm['conv1_1']
        self.conv1_1.bias = self.conv1_1.bias/self.norm_b['conv1_1']

        self.conv2.kernel = self.conv2.kernel/self.norm['conv2']
        self.conv2.bias = self.conv2.bias/self.norm_b['conv2']
        self.conv2_1.kernel = self.conv2_1.kernel/self.norm['conv2_1']
        self.conv2_1.bias = self.conv2_1.bias/self.norm_b['conv2_1']

        self.conv3.kernel = self.conv3.kernel/self.norm['conv3']
        self.conv3.bias = self.conv3.bias/self.norm_b['conv3']
        self.conv3_1.kernel = self.conv3_1.kernel/self.norm['conv3_1']
        self.conv3_1.bias = self.conv3_1.bias/self.norm_b['conv3_1']
        self.conv3_2.kernel = self.conv3_2.kernel/self.norm['conv3_2']
        self.conv3_2.bias = self.conv3_2.bias/self.norm_b['conv3_2']

        self.conv4.kernel = self.conv4.kernel/self.norm['conv4']
        self.conv4.bias = self.conv4.bias/self.norm_b['conv4']
        self.conv4_1.kernel = self.conv4_1.kernel/self.norm['conv4_1']
        self.conv4_1.bias = self.conv4_1.bias/self.norm_b['conv4_1']
        self.conv4_2.kernel = self.conv4_2.kernel/self.norm['conv4_2']
        self.conv4_2.bias = self.conv4_2.bias/self.norm_b['conv4_2']

        self.conv5.kernel = self.conv5.kernel/self.norm['conv5']
        self.conv5.bias = self.conv5.bias/self.norm_b['conv5']
        self.conv5_1.kernel = self.conv5_1.kernel/self.norm['conv5_1']
        self.conv5_1.bias = self.conv5_1.bias/self.norm_b['conv5_1']
        self.conv5_2.kernel = self.conv5_2.kernel/self.norm['conv5_2']
        self.conv5_2.bias = self.conv5_2.bias/self.norm_b['conv5_2']

        self.fc1.kernel = self.fc1.kernel/self.norm['fc1']
        self.fc1.bias = self.fc1.bias/self.norm_b['fc1']
        self.fc2.kernel = self.fc2.kernel/self.norm['fc2']
        self.fc2.bias = self.fc2.bias/self.norm_b['fc2']
        self.fc3.kernel = self.fc3.kernel/self.norm['fc3']
        self.fc3.bias = self.fc3.bias/self.norm_b['fc3']

    #
    def data_based_w_norm(self):

        ##f_new = False
        #f_new = True
        #if self.conf.model_name=='vgg_cifar100_ro_0':
            #f_new = False
#
        #if f_new:
            #path_stat=self.conf.path_stat
        #else:
            #path_stat='./stat/'

        path_stat=self.conf.path_stat

        f_name_stat_pre=self.conf.prefix_stat

        #f_name_stat='act_n_train'

        #stat_conf=['max','mean','max_999','max_99','max_98']

        f_stat=collections.OrderedDict()
        r_stat=collections.OrderedDict()

        stat='max_999'

        for idx_l, l in enumerate(self.layer_name):
            key=l+'_'+stat

            f_name_stat = f_name_stat_pre+'_'+key

            #if f_new:
            #    f_name=os.path.join(path_stat,f_name_stat)
            #    f_stat[key]=open(f_name,'r')
            #else:
            #    f_stat[key]=open(path_stat+f_name_stat+'_'+self.conf.model_name,'r')

            f_name=os.path.join(path_stat,f_name_stat)
            f_stat[key]=open(f_name,'r')

            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])


        self.w_norm_layer_wise()



    #
    def load_act_after_w_norm(self):
        #path_stat='./stat/'
        path_stat=self.conf.path_stat
        #f_name_stat='act_n_train_after_w_norm_max_999'

        f_stat=collections.OrderedDict()
        r_stat=collections.OrderedDict()

        # choose one
        stat='max_999'

        f_name_stat_pre=self.conf.prefix_stat

        for idx_l, l in enumerate(self.layer_name):
            key=l+'_'+stat

            f_name_stat = f_name_stat_pre+'_'+key

            f_name=os.path.join(path_stat,f_name_stat)
            f_stat[key]=open(f_name,'r')
            #f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])

                #print(self.dict_stat_r[l])


    def print_act_after_w_norm(self):
        self.load_act_after_w_norm()

        self.print_act_d()


    def temporal_norm(self):
        print('Temporal normalization')
        for key, value in self.list_layer.items():
            if self.conf.f_fused_bn:
                if not ('bn' in key):
                    value.kernel=value.kernel/self.tw
                    value.bias=value.bias/self.tw
            else:
                value.kernel=value.kernel/self.tw
                value.bias=value.bias/self.tw


    #
    def preproc_ann_norm(self):
        if self.conf.f_fused_bn:
            self.fused_bn()

        # weight normalization - data based
        if self.conf.f_w_norm_data:
            self.data_based_w_norm()


    def call_ann(self,inputs,f_training, tw=0, epoch=0):
        #print(type(inputs))
        #if self.f_1st_iter == False and self.conf.nn_mode=='ANN':
        if self.f_1st_iter == False:
            #assert False
            #if self.f_done_preproc == False:
                #self.f_done_preproc=True
                #self.print_model_conf()
                #self.preproc_ann_norm()
            self.f_skip_bn=self.conf.f_fused_bn
        else:
            self.f_skip_bn=False

        x = tf.reshape(inputs,self._input_shape)

        a_in = x

        s_conv1 = self.conv1(a_in)
        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.conv1_bn(s_conv1,training=f_training)

        a_conv1 = tf.nn.relu(s_conv1_bn)
        if f_training:
            a_conv1 = self.dropout_conv(a_conv1,training=f_training)
        s_conv1_1 = self.conv1_1(a_conv1)

        if self.f_skip_bn:
            s_conv1_1_bn = s_conv1_1
        else:
            s_conv1_1_bn = self.conv1_1_bn(s_conv1_1,training=f_training)
        a_conv1_1 = tf.nn.relu(s_conv1_1_bn)
        p_conv1_1 = self.pool2d(a_conv1_1)

        s_conv2 = self.conv2(p_conv1_1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.conv2_bn(s_conv2,training=f_training)
        a_conv2 = tf.nn.relu(s_conv2_bn)
        if f_training:
           a_conv2 = self.dropout_conv2(a_conv2,training=f_training)
        s_conv2_1 = self.conv2_1(a_conv2)
        if self.f_skip_bn:
            s_conv2_1_bn = s_conv2_1
        else:
            s_conv2_1_bn = self.conv2_1_bn(s_conv2_1,training=f_training)
        a_conv2_1 = tf.nn.relu(s_conv2_1_bn)
        p_conv2_1 = self.pool2d(a_conv2_1)

        s_conv3 = self.conv3(p_conv2_1)
        if self.f_skip_bn:
            s_conv3_bn = s_conv3
        else:
            s_conv3_bn = self.conv3_bn(s_conv3,training=f_training)
        a_conv3 = tf.nn.relu(s_conv3_bn)
        if f_training:
           a_conv3 = self.dropout_conv2(a_conv3,training=f_training)
        s_conv3_1 = self.conv3_1(a_conv3)
        if self.f_skip_bn:
            s_conv3_1_bn = s_conv3_1
        else:
            s_conv3_1_bn = self.conv3_1_bn(s_conv3_1,training=f_training)
        a_conv3_1 = tf.nn.relu(s_conv3_1_bn)
        if f_training:
           a_conv3_1 = self.dropout_conv2(a_conv3_1,training=f_training)
        s_conv3_2 = self.conv3_2(a_conv3_1)
        if self.f_skip_bn:
            s_conv3_2_bn = s_conv3_2
        else:
            s_conv3_2_bn = self.conv3_2_bn(s_conv3_2,training=f_training)
        a_conv3_2 = tf.nn.relu(s_conv3_2_bn)
        p_conv3_2 = self.pool2d(a_conv3_2)

        s_conv4 = self.conv4(p_conv3_2)
        if self.f_skip_bn:
            s_conv4_bn = s_conv4
        else:
            s_conv4_bn = self.conv4_bn(s_conv4,training=f_training)
        a_conv4 = tf.nn.relu(s_conv4_bn)
        if f_training:
           a_conv4 = self.dropout_conv2(a_conv4,training=f_training)
        s_conv4_1 = self.conv4_1(a_conv4)
        if self.f_skip_bn:
            s_conv4_1_bn = s_conv4_1
        else:
            s_conv4_1_bn = self.conv4_1_bn(s_conv4_1,training=f_training)
        a_conv4_1 = tf.nn.relu(s_conv4_1_bn)
        if f_training:
           a_conv4_1 = self.dropout_conv2(a_conv4_1,training=f_training)
        s_conv4_2 = self.conv4_2(a_conv4_1)
        if self.f_skip_bn:
            s_conv4_2_bn = s_conv4_2
        else:
            s_conv4_2_bn = self.conv4_2_bn(s_conv4_2,training=f_training)
        a_conv4_2 = tf.nn.relu(s_conv4_2_bn)
        p_conv4_2 = self.pool2d(a_conv4_2)

        s_conv5 = self.conv5(p_conv4_2)
        if self.f_skip_bn:
            s_conv5_bn = s_conv5
        else:
            s_conv5_bn = self.conv5_bn(s_conv5,training=f_training)
        a_conv5 = tf.nn.relu(s_conv5_bn)
        if f_training:
           a_conv5 = self.dropout_conv2(a_conv5,training=f_training)
        s_conv5_1 = self.conv5_1(a_conv5)
        if self.f_skip_bn:
            s_conv5_1_bn = s_conv5_1
        else:
            s_conv5_1_bn = self.conv5_1_bn(s_conv5_1,training=f_training)
        a_conv5_1 = tf.nn.relu(s_conv5_1_bn)
        if f_training:
           a_conv5_1 = self.dropout_conv2(a_conv5_1,training=f_training)
        s_conv5_2 = self.conv5_2(a_conv5_1)
        if self.f_skip_bn:
            s_conv5_2_bn = s_conv5_2
        else:
            s_conv5_2_bn = self.conv5_2_bn(s_conv5_2,training=f_training)
        a_conv5_2 = tf.nn.relu(s_conv5_2_bn)
        p_conv5_2 = self.pool2d(a_conv5_2)

        s_flat = tf.compat.v1.layers.flatten(p_conv5_2)

        if f_training:
           s_flat = self.dropout(s_flat,training=f_training)

        s_fc1 = self.fc1(s_flat)
        if self.f_skip_bn:
            s_fc1_bn = s_fc1
        else:
            s_fc1_bn = self.fc1_bn(s_fc1,training=f_training)
        a_fc1 = tf.nn.relu(s_fc1_bn)
        if f_training:
           a_fc1 = self.dropout(a_fc1,training=f_training)

        s_fc2 = self.fc2(a_fc1)
        if self.f_skip_bn:
            s_fc2_bn = s_fc2
        else:
            s_fc2_bn = self.fc2_bn(s_fc2,training=f_training)
        a_fc2 = tf.nn.relu(s_fc2_bn)
        if f_training:
           a_fc2 = self.dropout(a_fc2,training=f_training)

        s_fc3 = self.fc3(a_fc2)
        if self.f_skip_bn:
            s_fc3_bn = s_fc3
        else:
            s_fc3_bn = self.fc3_bn(s_fc3,training=f_training)

        a_fc3 = s_fc3_bn
        a_out = a_fc3


        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)


        if not self.f_1st_iter and (self.conf.f_train_time_const or self.conf.f_write_stat):
            #print("training time constant for temporal coding in SNN")

            self.dnn_act_list['in'] = a_in
            self.dnn_act_list['conv1']   = a_conv1
            self.dnn_act_list['conv1_1'] = a_conv1_1

            self.dnn_act_list['conv2']   = a_conv2
            self.dnn_act_list['conv2_1'] = a_conv2_1

            self.dnn_act_list['conv3']   = a_conv3
            self.dnn_act_list['conv3_1'] = a_conv3_1
            self.dnn_act_list['conv3_2'] = a_conv3_2

            self.dnn_act_list['conv4']   = a_conv4
            self.dnn_act_list['conv4_1'] = a_conv4_1
            self.dnn_act_list['conv4_2'] = a_conv4_2

            self.dnn_act_list['conv5']   = a_conv5
            self.dnn_act_list['conv5_1'] = a_conv5_1
            self.dnn_act_list['conv5_2'] = a_conv5_2

            self.dnn_act_list['fc1'] = a_fc1
            self.dnn_act_list['fc2'] = a_fc2
            self.dnn_act_list['fc3'] = a_fc3


        # write stat
        #print(self.f_1st_iter)
        #print(self.dict_stat_w["fc1"])
        if (self.conf.f_write_stat) and (not self.f_1st_iter):
            self.write_act()

            #self.dict_stat_w['conv1']=np.append(self.dict_stat_w['conv1'],a_conv1.numpy(),axis=0)
            #self.dict_stat_w['conv1_1']=np.append(self.dict_stat_w['conv1_1'],a_conv1_1.numpy(),axis=0)
            #self.dict_stat_w['conv2']=np.append(self.dict_stat_w['conv2'],a_conv2.numpy(),axis=0)
            #self.dict_stat_w['conv2_1']=np.append(self.dict_stat_w['conv2_1'],a_conv2_1.numpy(),axis=0)
            #self.dict_stat_w['conv3']=np.append(self.dict_stat_w['conv3'],a_conv3.numpy(),axis=0)
            #self.dict_stat_w['conv3_1']=np.append(self.dict_stat_w['conv3_1'],a_conv3_1.numpy(),axis=0)
            #self.dict_stat_w['conv3_2']=np.append(self.dict_stat_w['conv3_2'],a_conv3_2.numpy(),axis=0)
            #self.dict_stat_w['conv4']=np.append(self.dict_stat_w['conv4'],a_conv4.numpy(),axis=0)
            #self.dict_stat_w['conv4_1']=np.append(self.dict_stat_w['conv4_1'],a_conv4_1.numpy(),axis=0)
            #self.dict_stat_w['conv4_2']=np.append(self.dict_stat_w['conv4_2'],a_conv4_2.numpy(),axis=0)
            #self.dict_stat_w['conv5']=np.append(self.dict_stat_w['conv5'],a_conv5.numpy(),axis=0)
            #self.dict_stat_w['conv5_1']=np.append(self.dict_stat_w['conv5_1'],a_conv5_1.numpy(),axis=0)
            #self.dict_stat_w['conv5_2']=np.append(self.dict_stat_w['conv5_2'],a_conv5_2.numpy(),axis=0)
            #self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],a_fc1.numpy(),axis=0)
            #self.dict_stat_w['fc2']=np.append(self.dict_stat_w['fc2'],a_fc2.numpy(),axis=0)
            #self.dict_stat_w['fc3']=np.append(self.dict_stat_w['fc3'],a_fc3.numpy(),axis=0)

            # test bn activation distribution
            #self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],s_fc1_bn.numpy(),axis=0)

            #self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],s_fc1_bn.numpy(),axis=0)


        return a_out


    #
    def write_act(self):

        for l_name in self.layer_name_write_stat:
            dict_stat_w = self.dict_stat_w[l_name]

            if self.f_1st_iter_stat:
                dict_stat_w.assign(self.dnn_act_list[l_name])
            else:
                self.dict_stat_w[l_name]=tf.concat([dict_stat_w,self.dnn_act_list[l_name]], 0)


        #print(self.dict_stat_w[l_name].shape)

        if self.f_1st_iter_stat:
            self.f_1st_iter_stat = False


    #
    def print_model_conf(self):
        # print model configuration
        print('Input   N: '+str(self.in_shape))

        print('Conv1   S: '+str(self.conv1.kernel.get_shape()))
        print('Conv1   N: '+str(self.shape_out_conv1))
        print('Conv1_1 S: '+str(self.conv1_1.kernel.get_shape()))
        print('Conv1_1 N: '+str(self.shape_out_conv1_1))
        print('Pool1   N: '+str(self.shape_out_conv1_p))

        print('Conv2   S: '+str(self.conv2.kernel.get_shape()))
        print('Conv2   N: '+str(self.shape_out_conv2))
        print('Conv2_1 S: '+str(self.conv2_1.kernel.get_shape()))
        print('Conv2_1 N: '+str(self.shape_out_conv2_1))
        print('Pool2   N: '+str(self.shape_out_conv2_p))

        print('Conv3   S: '+str(self.conv3.kernel.get_shape()))
        print('Conv3   N: '+str(self.shape_out_conv3))
        print('Conv3_1 S: '+str(self.conv3_1.kernel.get_shape()))
        print('Conv3_1 N: '+str(self.shape_out_conv3_1))
        print('Conv3_2 S: '+str(self.conv3_2.kernel.get_shape()))
        print('Conv3_2 N: '+str(self.shape_out_conv3_2))
        print('Pool3   N: '+str(self.shape_out_conv3_p))

        print('Conv4   S: '+str(self.conv4.kernel.get_shape()))
        print('Conv4   N: '+str(self.shape_out_conv4))
        print('Conv4_1 S: '+str(self.conv4_1.kernel.get_shape()))
        print('Conv4_1 N: '+str(self.shape_out_conv4_1))
        print('Conv4_1 S: '+str(self.conv4_2.kernel.get_shape()))
        print('Conv4_2 N: '+str(self.shape_out_conv4_2))
        print('Pool4   N: '+str(self.shape_out_conv4_p))

        print('Conv5   S: '+str(self.conv5.kernel.get_shape()))
        print('Conv5   N: '+str(self.shape_out_conv5))
        print('Conv5_1 S: '+str(self.conv5_1.kernel.get_shape()))
        print('Conv5_1 N: '+str(self.shape_out_conv5_1))
        print('Conv5_1 S: '+str(self.conv5_1.kernel.get_shape()))
        print('Conv5_2 N: '+str(self.shape_out_conv5_2))
        print('Pool5   N: '+str(self.shape_out_conv5_p))

        print('Fc1     S: '+str(self.fc1.kernel.get_shape()))
        print('Fc1     N: '+str(self.shape_out_fc1))
        print('Fc2     S: '+str(self.fc2.kernel.get_shape()))
        print('Fc2     N: '+str(self.shape_out_fc2))
        print('Fc3     S: '+str(self.fc3.kernel.get_shape()))
        print('Fc3     N: '+str(self.shape_out_fc3))


    def print_act_d(self):
        print('print activation')

        fig, axs = plt.subplots(6,3)

        axs=axs.ravel()

        #for idx_l, (name_l,stat_l) in enumerate(self.dict_stat_r):
        for idx, (key, value) in enumerate(self.dict_stat_r.items()):
            axs[idx].hist(value.flatten())

        plt.show()


    def conv_bn_fused(self, conv, bn, time_step):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        conv.kernel = conv.kernel*math_ops.cast(inv,conv.kernel.dtype)
        conv.bias = ((conv.bias-mean)*inv+beta)


    def fc_bn_fused(self, conv, bn, time_step):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        conv.kernel = conv.kernel*math_ops.cast(inv,conv.kernel.dtype)
        conv.bias = ((conv.bias-mean)*inv+beta)


    def plot(self, x, y, mark):
        #plt.ion()
        #plt.hist(self.n_fc3.vmem)
        plt.plot(x, y, mark)
        plt.draw()
        plt.pause(0.00000001)
        #plt.ioff()


    def scatter(self, x, y, color, axe=None, marker='o'):
        if axe==None:
            plt.scatter(x, y, c=color, s=1, marker=marker)
            plt.draw()
            plt.pause(0.0000000000000001)
        else:
            axe.scatter(x, y, c=color, s=1, marker=marker)
            plt.draw()
            plt.pause(0.0000000000000001)

    def figure_hold(self):
        plt.close("dummy")
        plt.show()


    def visual(self, t):
        #plt.subplot2grid((2,4),(0,0))
        ax=plt.subplot(2,4,1)
        plt.title('w_sum_in (max)')
        self.plot(t,tf.reduce_max(s_fc2).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,1),sharex=ax)
        plt.subplot(2,4,2,sharex=ax)
        plt.title('vmem (max)')
        self.plot(t,tf.reduce_max(self.n_fc2.vmem).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,2))
        plt.subplot(2,4,3,sharex=ax)
        plt.title('# spikes (max)')
        self.plot(t,tf.reduce_max(self.n_fc2.get_spike_count()).numpy(), 'bo')
        #self.scatter(np.full(np.shape),tf.reduce_max(self.n_fc2.get_spike_count()).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,3))
        plt.subplot(2,4,4,sharex=ax)
        plt.title('spike neuron idx')
        plt.grid(True)
        plt.ylim([0,512])
        #plt.ylim([0,int(self.n_fc2.dim[1])])
        plt.xlim([0,tw])
        #self.plot(t,np.where(self.n_fc2.out.numpy()==1),'bo')
        #if np.where(self.n_fc2.out.numpy()==1).size == 0:
        idx_fire=np.where(self.n_fc2.out.numpy()==1)[1]
        if not len(idx_fire)==0:
            #print(np.shape(idx_fire))
            #print(idx_fire)
            #print(np.full(np.shape(idx_fire),t))
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')


        #plt.subplot2grid((2,4),(1,0))
        plt.subplot(2,4,5,sharex=ax)
        self.plot(t,tf.reduce_max(s_fc3).numpy(), 'bo')
        #plt.subplot2grid((2,4),(1,1))
        plt.subplot(2,4,6,sharex=ax)
        self.plot(t,tf.reduce_max(self.n_fc3.vmem).numpy(), 'bo')
        #plt.subplot2grid((2,4),(1,2))
        plt.subplot(2,4,7,sharex=ax)
        self.plot(t,tf.reduce_max(self.n_fc3.get_spike_count()).numpy(), 'bo')
        plt.subplot(2,4,8)
        plt.grid(True)
        #plt.ylim([0,self.n_fc3.dim[1]])
        plt.ylim([0,self.num_class])
        plt.xlim([0,tw])
        idx_fire=np.where(self.n_fc3.out.numpy()==1)[1]
        if not len(idx_fire)==0:
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')

    def get_total_residual_vmem(self):
        len=self.total_residual_vmem.shape[0]
        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            idx=idx_n-1
            if nn!='in' or nn!='fc3':
                self.total_residual_vmem[idx]+=tf.reduce_sum(tf.abs(n.vmem))
                self.total_residual_vmem[len-1]+=self.total_residual_vmem[idx]

    def get_total_isi(self):
        isi_count=np.zeros(self.conf.time_step)

        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc3':
                isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
                isi_count_n.resize(self.conf.time_step)
                isi_count = isi_count + isi_count_n

        return isi_count


    def f_out_isi(self,t):
        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc3':
                f_name = './isi/'+nn+'_'+self.conf.model_name+'_'+self.conf.input_spike_mode+'_'+self.conf.neural_coding+'_'+str(self.conf.time_step)+'.csv'

                if t==0:
                    f = open(f_name,'w')
                else:
                    f = open(f_name,'a')

                wr = csv.writer(f)

                array=n.isi.numpy().flatten()

                for i in range(len(array)):
                    if array[i]!=0:
                        wr.writerow((i,n.isi.numpy().flatten()[i]))

                f.close()


    def get_total_spike_amp(self):
        spike_amp=np.zeros(self.spike_amp_kind)
        #print(range(0,self.spike_amp_kind)[::-1])
        #print(np.power(0.5,range(0,self.spike_amp_kind)))

        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc3':
                spike_amp_n = np.histogram(n.out.numpy().flatten(),self.spike_amp_bin)
                #spike_amp = spike_amp + spike_amp_n[0]
                spike_amp += spike_amp_n[0]

                #isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
                #isi_count_n.resize(self.conf.time_step)
                #isi_count = isi_count + isi_count_n

        return spike_amp


    def get_total_spike_count(self):
        len=self.total_spike_count.shape[1]
        spike_count = np.zeros([len,])
        spike_count_int = np.zeros([len,])

        #for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            idx=idx_n-1
            if nn!='in':
                spike_count_int[idx]=tf.reduce_sum(n.get_spike_count_int())
                spike_count_int[len-1]+=spike_count_int[idx]
                spike_count[idx]=tf.reduce_sum(n.get_spike_count())
                spike_count[len-1]+=spike_count[idx]


        return [spike_count_int, spike_count]


    def bias_norm_proposed_method(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.bias = l.bias*self.conf.n_init_vth

    def bias_enable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = True

    def bias_disable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = False

    ###########################################
    # bias control
    ###########################################
    def bias_control(self,t):
        if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':
            #if self.conf.neural_coding == 'WEIGHTED_SPIKE':
            #if tf.equal(tf.reduce_max(a_in),0.0):
            if (int)(t%self.conf.p_ws) == 0:
                self.bias_enable()
            else:
                self.bias_disable()
        else:
            if self.conf.input_spike_mode == 'BURST':
                #TODO: check it
                self.bias_enable()
#                if t==0:
#                    self.bias_enable()
#                else:
#                    if tf.equal(tf.reduce_max(a_in),0.0):
#                        self.bias_enable()
#                    else:
#                        self.bias_disable()


        if self.conf.neural_coding == 'TEMPORAL':
            #if (int)(t%self.conf.p_ws) == 0:
            if t == 0:
                self.bias_enable()
            else:
                self.bias_disable()

    def bias_norm_weighted_spike(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
            #if (not 'bn' in k) and (not 'fc1' in k) :
                #l.bias = l.bias/(1-1/np.power(2,8))
                l.bias = l.bias/8.0

    def bias_norm_proposed_method(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.bias = l.bias*self.conf.n_init_vth
                #l.bias = l.bias/200
                #l.bias = l.bias*0.0

    def bias_enable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = True

    def bias_disable(self):
        for k, l in self.list_layer.items():
            if not 'bn' in k:
                l.use_bias = False

    def bias_restore(self):
        if self.conf.use_bias:
            self.bias_enable()
        else:
            self.bias_disable()


    ######################################################################
    # SNN call
    ######################################################################
    def call_snn(self,inputs,f_training,tw,epoch):

        #
        plt.clf()

        #
        for t in range(tw):
            if self.verbose == True:
                print('time: '+str(t))
            #x = tf.reshape(inputs,self._input_shape)

            self.bias_control(t)

            a_in = self.n_in(inputs,t)


            #if self.conf.f_real_value_input_snn:
            #    a_in = inputs
            #else:
            #    a_in = self.n_in(inputs,t)

            ####################
            # bias control
            ####################
#            if self.conf.input_spike_mode == 'WEIGHTED_SPIKE' or self.conf.neural_coding == 'WEIGHTED_SPIKE':
#                #if self.conf.neural_coding == 'WEIGHTED_SPIKE':
#                #if tf.equal(tf.reduce_max(a_in),0.0):
#                if (int)(t%self.conf.p_ws) == 0:
#                    self.bias_enable()
#                else:
#                    self.bias_disable()
#            else:
#                if self.conf.input_spike_mode == 'BURST':
#                    if t==0:
#                        self.bias_enable()
#                    else:
#                        if tf.equal(tf.reduce_max(a_in),0.0):
#                            self.bias_enable()
#                        else:
#                            self.bias_disable()
#
#
#            if self.conf.neural_coding == 'TEMPORAL':
#                #if (int)(t%self.conf.p_ws) == 0:
#                if t == 0:
#                    self.bias_enable()
#                else:
#                    self.bias_disable()


            ####################
            #
            ####################
            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)

            s_conv1_1 = self.conv1_1(a_conv1)
            a_conv1_1 = self.n_conv1_1(s_conv1_1,t)

            if self.conf.f_spike_max_pool:
                p_conv1_1 = lib_snn.spike_max_pool(
                    a_conv1_1,
                    self.n_conv1_1.get_spike_count(),
                    self.dict_shape['conv1_p']
                )
            else:
                p_conv1_1 = self.pool2d(a_conv1_1)

            s_conv2 = self.conv2(p_conv1_1)
            a_conv2 = self.n_conv2(s_conv2,t)
            s_conv2_1 = self.conv2_1(a_conv2)
            a_conv2_1 = self.n_conv2_1(s_conv2_1,t)

            if self.conf.f_spike_max_pool:
                p_conv2_1 = lib_snn.spike_max_pool(
                    a_conv2_1,
                    self.n_conv2_1.get_spike_count(),
                    self.dict_shape['conv2_p']
                )
            else:
                p_conv2_1 = self.pool2d(a_conv2_1)

            s_conv3 = self.conv3(p_conv2_1)
            a_conv3 = self.n_conv3(s_conv3,t)
            s_conv3_1 = self.conv3_1(a_conv3)
            a_conv3_1 = self.n_conv3_1(s_conv3_1,t)
            s_conv3_2 = self.conv3_2(a_conv3_1)
            a_conv3_2 = self.n_conv3_2(s_conv3_2,t)

            if self.conf.f_spike_max_pool:
                p_conv3_2 = lib_snn.spike_max_pool(
                    a_conv3_2,
                    self.n_conv3_2.get_spike_count(),
                    self.dict_shape['conv3_p']
                )
            else:
                p_conv3_2 = self.pool2d(a_conv3_2)


            s_conv4 = self.conv4(p_conv3_2)
            a_conv4 = self.n_conv4(s_conv4,t)
            s_conv4_1 = self.conv4_1(a_conv4)
            a_conv4_1 = self.n_conv4_1(s_conv4_1,t)
            s_conv4_2 = self.conv4_2(a_conv4_1)
            a_conv4_2 = self.n_conv4_2(s_conv4_2,t)

            if self.conf.f_spike_max_pool:
                p_conv4_2 = lib_snn.spike_max_pool(
                    a_conv4_2,
                    self.n_conv4_2.get_spike_count(),
                    self.dict_shape['conv4_p']
                )
            else:
                p_conv4_2 = self.pool2d(a_conv4_2)

            s_conv5 = self.conv5(p_conv4_2)
            a_conv5 = self.n_conv5(s_conv5,t)
            s_conv5_1 = self.conv5_1(a_conv5)
            a_conv5_1 = self.n_conv5_1(s_conv5_1,t)
            s_conv5_2 = self.conv5_2(a_conv5_1)
            a_conv5_2 = self.n_conv5_2(s_conv5_2,t)

            if self.conf.f_spike_max_pool:
                p_conv5_2 = lib_snn.spike_max_pool(
                    a_conv5_2,
                    self.n_conv5_2.get_spike_count(),
                    self.dict_shape['conv5_p']
                )
            else:
                p_conv5_2 = self.pool2d(a_conv5_2)

            flat = tf.compat.v1.layers.flatten(p_conv5_2)

            s_fc1 = self.fc1(flat)
            #s_fc1_bn = self.fc1_bn(s_fc1,training=f_training)
            #a_fc1 = self.n_fc1(s_fc1_bn,t)
            a_fc1 = self.n_fc1(s_fc1,t)

            s_fc2 = self.fc2(a_fc1)
            #s_fc2_bn = self.fc2_bn(s_fc2,training=f_training)
            #a_fc2 = self.n_fc2(s_fc2_bn,t)
            a_fc2 = self.n_fc2(s_fc2,t)

            s_fc3 = self.fc3(a_fc2)
            #print('a_fc3')
            a_fc3 = self.n_fc3(s_fc3,t)


            #print(str(t)+" : "+str(self.n_fc3.vmem.numpy()))



            ###
            #idx_print=0,10,10,0
            #print("time: {time:} - input: {input:0.5f}, vmem: {vmem:0.5f}, spk: {spike:f} {spike_bool:}\n"
                  #.format(
                        #time=t,
                        #input=inputs.numpy()[idx_print],
                        #vmem=self.n_conv1.vmem.numpy()[idx_print],
                        #spike=self.n_conv1.out.numpy()[idx_print],
                        #spike_bool= (self.n_conv1.out.numpy()[idx_print]>0.0))
                  #)



            if self.f_1st_iter == False and self.f_debug_visual == True:
                #self.visual(t)

                synapse=s_conv1
                neuron=self.n_in

                #synapse=s_fc2
                #neuron=self.n_fc2

                synapse_1=s_conv1
                neuron_1=self.n_conv1

                synapse_2 = s_conv1_1
                neuron_2 = self.n_conv1_1


                #self.debug_visual(synapse, neuron, synapse_1, neuron_1, synapse_2, neuron_2, t)
                self.debug_visual_raster(t)

                #if t==self.tw-1:
                #    plt.figure()
                #    #plt.show()

            ##########
            #
            ##########

            if self.f_1st_iter == False:

                if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                    output=self.n_fc3.vmem
                    self.recoding_ret_val()


                    #num_spike_count = tf.cast(tf.reduce_sum(self.spike_count,axis=[2]),tf.int32)
                    #num_spike_count = tf.reduce_sum(self.spike_count,axis=[2])

            #print(t, self.n_fc3.last_spike_time.numpy())
            #print(t, self.n_fc3.isi.numpy())


        if self.f_1st_iter:
            self.f_1st_iter = False

            self.conv1_bn(s_conv1,training=f_training)
            self.conv1_1_bn(s_conv1_1,training=f_training)

            self.conv2_bn(s_conv2,training=f_training)
            self.conv2_1_bn(s_conv2_1,training=f_training)

            self.conv3_bn(s_conv3,training=f_training)
            self.conv3_1_bn(s_conv3_1,training=f_training)
            self.conv3_2_bn(s_conv3_2,training=f_training)

            self.conv4_bn(s_conv4,training=f_training)
            self.conv4_1_bn(s_conv4_1,training=f_training)
            self.conv4_2_bn(s_conv4_2,training=f_training)

            self.conv5_bn(s_conv5,training=f_training)
            self.conv5_1_bn(s_conv5_1,training=f_training)
            self.conv5_2_bn(s_conv5_2,training=f_training)

            self.fc1_bn(s_fc1,training=f_training)
            self.fc2_bn(s_fc2,training=f_training)
            self.fc3_bn(s_fc3,training=f_training)

            return 0


        else:

            self.get_total_residual_vmem()

            #spike_zero = tf.reduce_sum(self.spike_count,axis=[0,2])
            spike_zero = tf.reduce_sum(self.snn_output,axis=[0,2])
            #spike_zero = tf.count_nonzero(self.spike_count,axis=[0,2])

            if np.any(spike_zero.numpy() == 0.0):
            #if num_spike_count==0.0:
                print('spike count 0')
                #print(num_spike_count.numpy())
                #a = input("press any key to exit")
                #os.system("Pause")
                #raw_input("Press any key to exit")
                #sys.exit(0)


            # first_spike_time visualization
            if self.conf.f_record_first_spike_time and self.conf.f_visual_record_first_spike_time:
                print('first spike time')
                _, axes = plt.subplots(4,4)
                idx_plot=0
                for n_name, n in self.list_neuron.items():
                    if not ('fc3' in n_name):
                        #positive = n.first_spike_time > 0
                        #print(n_name+'] min: '+str(tf.reduce_min(n.first_spike_time[positive]))+', mean: '+str(tf.reduce_mean(n.first_spike_time[positive])))
                        #print(tf.reduce_min(n.first_spike_time[positive]))

                        #positive=n.first_spike_time.numpy().flatten() > 0
                        positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0)

                        if not tf.equal(tf.size(positive),0):

                            #min=np.min(n.first_spike_time.numpy().flatten()[positive,])
                            #print(positive.shape)
                            #min=np.min(n.first_spike_time.numpy().flatten()[positive])
                            min=tf.reduce_min(positive)
                            #mean=np.mean(n.first_spike_time.numpy().flatten()[positive,])

                            #if self.conf.f_tc_based:
                            #    fire_s=idx_plot*self.conf.time_fire_start
                            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration
                            #else:
                            #    fire_s=idx_plot*self.conf.time_fire_start
                            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration

                            #fire_s = n.time_start_fire
                            #fire_e = n.time_end_fire

                            fire_s = idx_plot * self.conf.time_fire_start
                            fire_e = idx_plot * self.conf.time_fire_start + self.conf.time_fire_duration

                            axe=axes.flatten()[idx_plot]
                            #axe.hist(n.first_spike_time.numpy().flatten()[positive],bins=range(fire_s,fire_e,1))
                            axe.hist(positive.numpy().flatten(),bins=range(fire_s,fire_e,1))

                            axe.axvline(x=min.numpy(),color='b', linestyle='dashed')

                            axe.axvline(x=fire_s)
                            axe.axvline(x=fire_e)


                        idx_plot+=1



                # file write raw data
                for n_name, n in self.list_neuron.items():
                    if not ('fc3' in n_name):
                        positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0).numpy()

                        fname = './spike_time/spike-time'
                        if self.conf.f_load_time_const:
                            fname += '_train-'+str(self.conf.time_const_num_trained_data)+'_tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.time_window)

                        fname += '_'+n_name+'.csv'
                        f = open(fname,'w')
                        wr = csv.writer(f)
                        wr.writerow(positive)
                        f.close()


                plt.show()


        #return self.spike_count
        return self.snn_output


    ###########################################################
    ## SNN output
    ###########################################################

    #
    def recoding_ret_val(self):
        output=self.snn_output_func()
        self.snn_output.scatter_nd_update([self.count_accuracy_time_point],tf.expand_dims(output,0))

        tc_int, tc = self.get_total_spike_count()
        self.total_spike_count_int[self.count_accuracy_time_point]+=tc_int
        self.total_spike_count[self.count_accuracy_time_point]+=tc

        self.count_accuracy_time_point+=1

        #num_spike_count = tf.cast(tf.reduce_sum(self.snn_output,axis=[2]),tf.int32)

    def snn_output_func(self):
        snn_output_func_sel = {
            "SPIKE": self.snn_output_layer.spike_counter,
            "VMEM": self.snn_output_layer.vmem,
            "FIRST_SPIKE_TIME": self.snn_output_layer.first_spike_time
        }
        return snn_output_func_sel[self.conf.snn_output_type]




    def reset_neuron(self):
        for idx, l in self.list_neuron.items():
            l.reset()

    def debug_visual(self, synapse, neuron, synapse_1, neuron_1, synapse_2, neuron_2, t):

        idx_print_s, idx_print_e = 0, 200

        ax=plt.subplot(3,4,1)
        plt.title('w_sum_in (max)')
        self.plot(t,tf.reduce_max(tf.reshape(synapse,[-1])).numpy(), 'bo')
        #plt.subplot2grid((2,4),(0,1),sharex=ax)
        plt.subplot(3,4,2,sharex=ax)
        plt.title('vmem (max)')
        #self.plot(t,tf.reduce_max(neuron.vmem).numpy(), 'bo')
        self.plot(t,tf.reduce_max(tf.reshape(neuron.vmem,[-1])).numpy(), 'bo')
        self.plot(t,tf.reduce_max(tf.reshape(neuron.vth,[-1])).numpy(), 'ro')
        #self.plot(t,neuron.out.numpy()[neuron.out.numpy()>0].sum(), 'bo')
        #plt.subplot2grid((2,4),(0,2))
        plt.subplot(3,4,3,sharex=ax)
        plt.title('# spikes (total)')
        #spike_rate=neuron.get_spike_count()/t
        self.plot(t,tf.reduce_sum(tf.reshape(neuron.spike_counter_int,[-1])).numpy(), 'bo')
        #self.plot(t,neuron.vmem.numpy()[neuron.vmem.numpy()>0].sum(), 'bo')
        #self.plot(t,tf.reduce_max(spike_rate), 'bo')
        #plt.subplot2grid((2,4),(0,3))
        plt.subplot(3,4,4,sharex=ax)
        plt.title('spike neuron idx')
        plt.grid(True)
        #plt.ylim([0,512])
        plt.ylim([0,neuron.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
        #plt.ylim([0,int(self.n_fc2.dim[1])])
        plt.xlim([0,self.tw])
        #self.plot(t,np.where(self.n_fc2.out.numpy()==1),'bo')
        #if np.where(self.n_fc2.out.numpy()==1).size == 0:
        idx_fire=np.where(neuron.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
        if not len(idx_fire)==0:
            #print(np.shape(idx_fire))
            #print(idx_fire)
            #print(np.full(np.shape(idx_fire),t))
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')
            #self.scatter(t,np.argmax(neuron.get_spike_count().numpy().flatten()),'b')

        addr=0,0,0,6
        ax=plt.subplot(3,4,5)
        #self.plot(t,tf.reduce_max(tf.reshape(synapse_1,[-1])).numpy(), 'bo')
        self.plot(t,synapse_1[addr].numpy(), 'bo')
        plt.subplot(3,4,6,sharex=ax)
        #self.plot(t,tf.reduce_max(tf.reshape(neuron_1.vmem,[-1])).numpy(), 'bo')
        self.plot(t,neuron_1.vmem.numpy()[addr], 'bo')
        self.plot(t,neuron_1.vth.numpy()[addr], 'ro')
        plt.subplot(3,4,7,sharex=ax)
        #self.plot(t,neuron_1.vmem.numpy()[neuron_1.vmem.numpy()>0].sum(), 'bo')
        self.plot(t,neuron_1.spike_counter_int.numpy()[addr], 'bo')
        plt.subplot(3,4,8,sharex=ax)
        plt.grid(True)
        plt.ylim([0,neuron_1.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
        plt.xlim([0,self.tw])
        idx_fire=np.where(neuron_1.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
        #idx_fire=neuron_1.f_fire.numpy().flatten()[idx_print_s:idx_print_e]
        #idx_fire=neuron_1.f_fire.numpy()[0,0,0,0:10]
        #print(neuron_1.vmem.numpy()[0,0,0,1])
        #print(neuron_1.f_fire.numpy()[0,0,0,1])
        #print(idx_fire)
        if not len(idx_fire)==0:
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')

        addr=0,0,0,6
        ax=plt.subplot(3,4,9)
        #self.plot(t,tf.reduce_max(tf.reshape(synapse_2,[-1])).numpy(), 'bo')
        self.plot(t,synapse_2[addr].numpy(), 'bo')
        plt.subplot(3,4,10,sharex=ax)
        #self.plot(t,tf.reduce_max(tf.reshape(neuron_2.vmem,[-1])).numpy(), 'bo')
        self.plot(t,neuron_2.vmem.numpy()[addr], 'bo')
        self.plot(t,neuron_2.vth.numpy()[addr], 'ro')
        plt.subplot(3,4,11,sharex=ax)
        #self.plot(t,neuron_2.vmem.numpy()[neuron_2.vmem.numpy()>0].sum(), 'bo')
        self.plot(t,neuron_2.spike_counter_int.numpy()[addr], 'bo')
        plt.subplot(3,4,12,sharex=ax)
        plt.grid(True)
        plt.ylim([0,neuron_2.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
        plt.xlim([0,self.tw])
        idx_fire=np.where(neuron_2.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
        #idx_fire=neuron_2.f_fire.numpy().flatten()[idx_print_s:idx_print_e]
        #idx_fire=neuron_2.f_fire.numpy()[0,0,0,0:10]
        #print(neuron_2.vmem.numpy()[0,0,0,1])
        #print(neuron_2.f_fire.numpy()[0,0,0,1])
        #print(idx_fire)
        if not len(idx_fire)==0:
            self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')




    #@tfplot.autowrap
    def debug_visual_raster(self,t):

        subplot_x, subplot_y = 4, 4

        num_subplot = subplot_x*subplot_y
        idx_print_s, idx_print_e = 0, 100

        if t==0:
            plt_idx=0
            #plt.figure()
            plt.close("dummy")
            _, self.debug_visual_axes = plt.subplots(subplot_y,subplot_x)

            for neuron_name, neuron in self.list_neuron.items():
                if not ('fc3' in neuron_name):
                    self.debug_visual_list_neuron[neuron_name]=neuron

                    axe = self.debug_visual_axes.flatten()[plt_idx]

                    axe.set_ylim([0,neuron.out.numpy().flatten()[idx_print_s:idx_print_e].size])
                    axe.set_xlim([0,self.tw])

                    axe.grid(True)

                    if(plt_idx>0):
                        axe.axvline(x=(plt_idx-1)*self.conf.time_fire_start, color='b')                 # integration
                    axe.axvline(x=plt_idx*self.conf.time_fire_start, color='b')                         # fire start
                    axe.axvline(x=plt_idx*self.conf.time_fire_start+self.conf.time_fire_duration, color='b') # fire end

                    plt_idx+=1

        else:
            plt_idx=0
            for neuron_name, neuron in self.debug_visual_list_neuron.items():

                t_fire_s = plt_idx*self.conf.time_fire_start
                t_fire_e = t_fire_s + self.conf.time_fire_duration

                if t >= t_fire_s and t < t_fire_e :
                    if tf.reduce_sum(neuron.out) != 0.0:
                        idx_fire=tf.where(tf.not_equal(tf.reshape(neuron.out,[-1])[idx_print_s:idx_print_e],tf.constant(0,dtype=tf.float32)))

                    #if tf.size(idx_fire) != 0:
                        axe = self.debug_visual_axes.flatten()[plt_idx]
                        self.scatter(tf.fill(idx_fire.shape,t),idx_fire,'r', axe=axe)

                plt_idx+=1

        if t==self.tw-1:
            plt.figure("dummy")


    # training time constant for temporal coding
    def train_time_const(self):

        print("models: train_time_const")

        # train_time_const
        name_layer_prev=''
        for name_layer, layer in self.list_neuron.items():
            if not ('fc3' in name_layer):
                dnn_act = self.dnn_act_list[name_layer]
                self.list_neuron[name_layer].train_time_const_fire(dnn_act)

            if not ('in' in name_layer):
                self.list_neuron[name_layer].set_time_const_integ(self.list_neuron[name_layer_prev].time_const_fire)

            name_layer_prev = name_layer


        # train_time_delay
        name_layer_prev=''
        for name_layer, layer in self.list_neuron.items():
            if not ('fc3' in name_layer or 'in' in name_layer):
                dnn_act = self.dnn_act_list[name_layer]
                self.list_neuron[name_layer].train_time_delay_fire(dnn_act)

            if not ('in' in name_layer or 'conv1' in name_layer):
                self.list_neuron[name_layer].set_time_delay_integ(self.list_neuron[name_layer_prev].time_delay_fire)

            name_layer_prev = name_layer


    def get_time_const_train_loss(self):

        loss_prec=0
        loss_min=0
        loss_max=0

        for name_layer, layer in self.list_neuron.items():
            if not ('fc3' in name_layer):
                loss_prec += self.list_neuron[name_layer].loss_prec
                loss_min += self.list_neuron[name_layer].loss_min
                loss_max += self.list_neuron[name_layer].loss_max

        return [loss_prec, loss_min, loss_max]



    ##############################################################
    # save activation for data-based normalization
    ##############################################################
    # distribution of activation - neuron-wise or channel-wise?
    #def save_dist_activation_neuron_vgg16(model):
    def save_activation(self):

        path_stat=self.conf.path_stat
        f_name_stat_pre=self.conf.prefix_stat


        stat_conf=['max_999']
        f_stat=collections.OrderedDict()

        #
        threads=[]

        for idx_l, l in enumerate(self.layer_name_write_stat):
            for idx_c, c in enumerate(stat_conf):
                key=l+'_'+c

                f_name_stat = f_name_stat_pre+'_'+key
                f_name=os.path.join(path_stat,f_name_stat)
                #f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'w')
                #f_stat[key]=open(path_stat'/'f_name_stat)
                #print(f_name)

                f_stat[key]=open(f_name,'w')
                #wr_stat[key]=csv.writer(f_stat[key])


                #for idx_l, l in enumerate(self.layer_name_write_stat):
                threads.append(threading.Thread(target=self.write_stat, args=(f_stat[key], l, c)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        #f = open('./stat/dist_act_neuron_trainset_test_'+stat_conf+conf.model_name,'wb')

        #wr = csv.writer(f,quoting=csv.QUOTE_NONE,escapechar='\n')
        #wr = csv.writer(f)
        #wr.writerow(['max','min','mean','std','99.9','99','98'])
        #wr.writerow(['max','min','mean','99.9','99','98'])

#        for idx_l, l in enumerate(self.layer_name_write_stat):
#            s_layer=self.dict_stat_w[l].numpy()
#            #print(l)
#            #print(np.shape(s_layer))
#            #print(np.shape(s_layer)[1:])
#            #shape_n=np.shape(s_layer)[1:]
#
#            # before
#            #max=np.max(s_layer,axis=0).flatten()
#            #mean=np.mean(s_layer,axis=0).flatten()
#            #max_999=np.nanpercentile(s_layer,99.9,axis=0).flatten()
#            max_999=np.nanpercentile(s_layer,99.9,axis=0).flatten()
#            #max_99=np.nanpercentile(s_layer,99,axis=0).flatten()
#            #max_98=np.nanpercentile(s_layer,98,axis=0).flatten()
#            #max_95=np.nanpercentile(s_layer,95,axis=0).flatten()
#            #max_90=np.nanpercentile(s_layer,90,axis=0).flatten()
#            #wr_stat[l+'_max'].writerow(max)
#            #wr_stat[l+'_mean'].writerow(mean)
#            wr_stat[l+'_max_999'].writerow(max_999)
#            #wr_stat[l+'_max_99'].writerow(max_99)
#            #wr_stat[l+'_max_98'].writerow(max_98)
#            #wr_stat[l+'_max_95'].writerow(max_95)
#            #wr_stat[l+'_max_90'].writerow(max_90)
#
#
#            #min=np.min(s_layer,axis=0).flatten()
#            #max=np.max(s_layer).flatten()
#            #print(max)
#            #print(np.nanpercentile(s_layer,99.9))
#            #print(np.nanpercentile(s_layer,99))
#            #print(np.nanpercentile(s_layer,98))
#            #print(np.nanpercentile(s_layer,95))
#            #print(np.nanpercentile(s_layer,90))
#            #print(np.mean(s_layer))
#            #plt.hist(s_layer.flatten(), log=True, bins=int(max*2))
#            #plt.show()
#
#            # for after w norm stat
#            #max=np.max(s_layer,axis=0).flatten()
#            #mean=np.mean(s_layer,axis=0).flatten()
#            #min=np.mean(s_layer,axis=0).flatten()
#            #max_25=np.nanpercentile(s_layer,25,axis=0).flatten()
#            #max_75=np.nanpercentile(s_layer,75,axis=0).flatten()
#            #hist=np.histogram(s_layer)
#
#            #wr_stat[l+'_max'].writerow(max)
#            #wr_stat[l+'_mean'].writerow(mean)
#            #wr_stat[l+'_min'].writerow(min)
#            #wr_stat[l+'_max_75'].writerow(max_25)
#            #wr_stat[l+'_max_25'].writerow(max_75)
#            #wr_stat[l+'_hist'].writerow()
#
#
#            #print(np.shape(max))
#
#            #np.savetxt('a',max)
#
#            #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.std(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
#            #wr.writerow([np.max(s_layer,axis=0),np.min(s_layer,axis=0),np.mean(s_layer,axis=0),np.nanpercentile(s_layer,99.9,axis=0),np.nanpercentile(s_layer,99,axis=0),np.nanpercentile(s_layer,98,axis=0)])
#            #wr.writerow([max,min,mean,max_999,max_99,max_98])
#
#        for idx_l, l in enumerate(self.layer_name_write_stat):
#            for idx_c, c in enumerate(stat_conf):
#                key=l+'_'+c
#                f_stat[key].close()




    def write_stat(self, f_stat, layer_name, stat_conf_name):
        print('write_stat func')

        l = layer_name
        c = stat_conf_name
        s_layer=self.dict_stat_w[l].numpy()

        self._write_stat(f_stat,s_layer,c)


        #f_stat.close()


    def _write_stat(self, f_stat, s_layer, conf_name):
        print('stat cal: '+conf_name)

        if conf_name=='max':
            stat=np.max(s_layer,axis=0).flatten()
            #stat=tf.reshape(tf.reduce_max(s_layer,axis=0),[-1])
        elif conf_name=='max_999':
            stat=np.nanpercentile(s_layer,99.9,axis=0).flatten()
        elif conf_name=='max_99':
            stat=np.nanpercentile(s_layer,99,axis=0).flatten()
        elif conf_name=='max_98':
            stat=np.nanpercentile(s_layer,98,axis=0).flatten()
        else:
            print('stat confiugration not supported')

        print('stat write')
        wr_stat=csv.writer(f_stat)
        wr_stat.writerow(stat)
        f_stat.close()






    ##############################################################
    def plot_dist_activation_vgg16(self):
    #        plt.subplot2grid((6,3),(0,0))
    #        plt.hist(model.stat_a_conv1)
    #        plt.subplot2grid((6,3),(0,1))
    #        plt.hist(model.stat_a_conv1_1)
    #        plt.subplot2grid((6,3),(1,0))
    #        plt.hist(model.stat_a_conv2)
    #        plt.subplot2grid((6,3),(1,1))
    #        plt.hist(model.stat_a_conv2_1)
    #        plt.subplot2grid((6,3),(2,0))
    #        plt.hist(model.stat_a_conv3)
    #        plt.subplot2grid((6,3),(2,1))
    #        plt.hist(model.stat_a_conv3_1)
    #        plt.subplot2grid((6,3),(2,2))
    #        plt.hist(model.stat_a_conv3_1)
    #        plt.subplot2grid((6,3),(3,0))
    #        plt.hist(model.stat_a_conv4)
    #        plt.subplot2grid((6,3),(3,1))
    #        plt.hist(model.stat_a_conv4_1)
    #        plt.subplot2grid((6,3),(3,2))
    #        plt.hist(model.stat_a_conv4_2)
    #        plt.subplot2grid((6,3),(4,0))
    #        plt.hist(model.stat_a_conv5)
    #        plt.subplot2grid((6,3),(4,1))
    #        plt.hist(model.stat_a_conv5_1)
    #        plt.subplot2grid((6,3),(4,2))
    #        plt.hist(model.stat_a_conv5_2)
    #        plt.subplot2grid((6,3),(5,0))
    #        plt.hist(model.stat_a_fc1)
    #        plt.subplot2grid((6,3),(5,1))
    #        plt.hist(model.stat_a_fc2)
    #        plt.subplot2grid((6,3),(5,2))
    #        plt.hist(model.stat_a_fc3)

        print(np.shape(self.dict_stat_w['fc1']))

        plt.hist(self.dict_stat_w["fc1"][:,120],bins=100)
        plt.show()








# distribution of activation - layer-wise
def save_dist_activation_vgg16(model):
    layer_name=[
        #model.stat_a_conv1,
        #model.stat_a_conv1_1
        #model.stat_a_conv2,
        #model.stat_a_conv2_1,
        #model.stat_a_conv3,
        #model.stat_a_conv3_1,
        #model.stat_a_conv3_2,
        #model.stat_a_conv4,
        #model.stat_a_conv4_1,
        #model.stat_a_conv4_2,
        #model.stat_a_conv5,
        #model.stat_a_conv5_1,
        #model.stat_a_conv5_2,
        #model.stat_a_fc1,
        #model.stat_a_fc2,
        model.stat_a_fc3
    ]

    f = open('./stat/dist_act_neuron_trainset_fc_'+conf.model_name,'wb')
    wr = csv.writer(f)
    wr.writerow(['max','min','mean','std','99.9','99','98'])

    for _, s_layer in enumerate(layer_name):
        wr.writerow([np.max(s_layer),np.min(s_layer),np.mean(s_layer),np.std(s_layer),np.nanpercentile(s_layer,99.9),np.nanpercentile(s_layer,99),np.nanpercentile(s_layer,98)])
    f.close()













