import tensorflow as tf
#import tensorflow.contrib.eager as tfe

#from tensorflow.contrib.layers.python.layers import initializers
#from tensorflow.contrib.layers.python.layers import regularizers

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops

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

from collections import OrderedDict

from scipy import stats

#
#class MNISTModel_CNN(tfe.Network):
#class MNISTModel_CNN(tf.keras.layers):
class MNISTModel_CNN(tf.keras.layers.Layer):
    def __init__(self, data_format, conf):
        super(MNISTModel_CNN, self).__init__(name='')

        self.data_format = data_format
        self.conf = conf
        self.num_class = self.conf.num_class

        self.f_1st_iter = True
        self.verbose = conf.verbose
        self.f_debug_visual = conf.verbose_visual

        #
        self.epoch = -1

        if self.conf.en_train:
            self.f_done_preproc = True
        else:
            self.f_done_preproc = False

        self.f_done_postproc = False


        self.kernel_size = 5
        self.fanin_conv = self.kernel_size*self.kernel_size
        #self.fanin_conv = self.kernel_size*self.kernel_size/9


        self.tw=conf.time_step


        self.count_accuracy_time_point=0
        self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        self.accuracy_time_point.append(conf.time_step)
        #self.num_accuracy_time_point = int(math.ceil(float(conf.time_step)/float(conf.time_step_save_interval))
        self.num_accuracy_time_point = len(self.accuracy_time_point)


        #
        self.f_skip_bn = self.conf.f_fused_bn

        self.layer_name=[
            'in',
            'conv1',
            'conv2',
            'fc1',
        ]

        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])

        self.total_residual_vmem=np.zeros(len(self.layer_name)+1)

        if self.conf.f_isi:
            self.total_isi=np.zeros(self.conf.time_step)

            type_spike_amp_kind = {
                'RATE': 1,
                'WEIGHTED_SPIKE': self.conf.p_ws,
                'PROPOSED': int(5-np.log2(self.conf.n_init_vth))
            }

            self.spike_amp_kind = type_spike_amp_kind[self.conf.neural_coding]+1
            self.total_spike_amp=np.zeros(self.spike_amp_kind)

            type_spike_amp_bin = {
                'RATE': np.power(0.5,range(0,self.spike_amp_kind+1)),
                'WEIGHTED_SPIKE': np.power(0.5,range(0,self.spike_amp_kind+1)),
                'PROPOSED': 32*np.power(0.5,range(0,self.spike_amp_kind+1))
            }

            self.spike_amp_bin=type_spike_amp_bin[self.conf.neural_coding]
            self.spike_amp_bin=self.spike_amp_bin[::-1]
            self.spike_amp_bin[0]=0.0

        if self.conf.f_comp_act:
            self.total_comp_act=np.zeros([self.conf.time_step,len(self.layer_name)+1])


        self.output_layer_isi=np.zeros(self.num_class)
        self.output_layer_last_spike_time=np.zeros(self.num_class)

        # nomarlization factor
        self.norm=collections.OrderedDict()
        self.norm_b=collections.OrderedDict()


        if self.data_format == 'channels_first':
            self._input_shape = [-1,1,28,28]   # MNIST
        else:
            assert self.data_format == 'channels_last'
            self._input_shape = [-1,28,28,1]


        #
        use_bias = conf.use_bias


        activation = None
        padding = 'same'

# TODO: TF-V1
#        regularizer_type = {
#            'L1': regularizers.l1_regularizer(conf.lamb),
#            'L2': regularizers.l2_regularizer(conf.lamb),
#            'L1_L2': regularizers.l1_l2_regularizer(conf.lamb,conf.lamb)
#        }

        regularizer_type = {
            'L1': regularizers.l1(conf.lamb),
            'L2': regularizers.l2(conf.lamb),
            'L1_L2': regularizers.l1_l2(conf.lamb,conf.lamb)
        }



        kernel_regularizer = regularizer_type[self.conf.regularizer]
        #kernel_initializer = initializers.xavier_initializer(True)
        #kernel_initializer = initializers.GlorotNormal()
        kernel_initializer = initializers.GlorotUniform()
        #kernel_initializer = initializers.variance_scaling_initializer(factor=2.0,mode='FAN_IN')    # MSRA init. = He init


        self.list_layer=collections.OrderedDict()
        # TODO: TF-V1
        #self.list_layer['conv1'] = self.track_layer(tf.layers.Conv2D(12,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='valid'))
        #self.list_layer['conv1_bn'] = self.track_layer(tf.layers.BatchNormalization())
        #self.list_layer['conv2'] = self.track_layer(tf.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='valid'))
        #self.list_layer['conv2_bn'] = self.track_layer(tf.layers.BatchNormalization())
        #self.list_layer['fc1'] = self.track_layer(tf.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer))
        #self.dropout = self.track_layer(tf.layers.Dropout(0.5))

        self.list_layer['conv1'] = tf.keras.layers.Conv2D(12,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='valid')
        self.list_layer['conv1_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['conv2'] = tf.keras.layers.Conv2D(64,self.kernel_size,data_format=data_format,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer,padding='valid')
        self.list_layer['conv2_bn'] = tf.keras.layers.BatchNormalization()
        self.list_layer['fc1'] = tf.keras.layers.Dense(self.num_class,use_bias=use_bias,kernel_regularizer=kernel_regularizer,kernel_initializer=kernel_initializer)
        self.dropout = tf.keras.layers.Dropout(0.5)


        # remove later
        self.conv1=self.list_layer['conv1']
        self.conv1_bn=self.list_layer['conv1_bn']
        self.conv2=self.list_layer['conv2']
        self.conv2_bn=self.list_layer['conv2_bn']
        self.fc1=self.list_layer['fc1']

        self.pool2d = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME', data_format=data_format)
        self.act_relu = tf.nn.relu

        input_shape_one_sample = tensor_shape.TensorShape([1,self._input_shape[1],self._input_shape[2],self._input_shape[3]])
        self.in_shape = [self.conf.batch_size]+self._input_shape[1:]

        self.shape_out_conv1 = util.cal_output_shape_Conv2D_pad_val(self.data_format,self.in_shape,12,self.kernel_size,1)
        self.shape_out_conv1_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv1,2,2)

        self.shape_out_conv2 = util.cal_output_shape_Conv2D_pad_val(self.data_format,self.shape_out_conv1_p,64,self.kernel_size,1)
        self.shape_out_conv2_p = util.cal_output_shape_Pooling2D(self.data_format,self.shape_out_conv2,2,2)

        self.shape_out_fc1 = tensor_shape.TensorShape([self.conf.batch_size,self.num_class]).as_list()


        self.dict_shape=collections.OrderedDict()
        #self.dict_shape['conv1']=self.shape_out_conv1
        #self.dict_shape['conv1_p']=self.shape_out_conv1_p
        #self.dict_shape['conv2']=self.shape_out_conv2
        #self.dict_shape['conv2_p']=self.shape_out_conv2_p
        #self.dict_shape['fc1']=self.shape_out_fc1
        self.dict_shape['conv1']=self.shape_out_conv1
        self.dict_shape['conv1_p']=self.shape_out_conv1_p
        self.dict_shape['conv2']=self.shape_out_conv2
        self.dict_shape['conv2_p']=self.shape_out_conv2_p
        self.dict_shape['fc1']=self.shape_out_fc1


        self.dict_shape_one_batch=collections.OrderedDict()
        self.dict_shape_one_batch['in']= [1,] + self._input_shape[1:]
        self.dict_shape_one_batch['conv1']=[1,]+self.shape_out_conv1[1:]
        self.dict_shape_one_batch['conv1_p']=[1,]+self.shape_out_conv1_p[1:]
        self.dict_shape_one_batch['conv2']=[1,]+self.shape_out_conv2[1:]
        self.dict_shape_one_batch['conv2_p']=[1,]+self.shape_out_conv2_p[1:]
        self.dict_shape_one_batch['fc1']=[1,]+self.shape_out_fc1[1:]

        #
        self.dict_stat_r=collections.OrderedDict()  # read
        self.dict_stat_w=collections.OrderedDict()  # write


        if self.conf.f_entropy:
            self.dict_stat_w['conv1']=np.zeros([self.conf.time_step,]+self.shape_out_conv1[1:])
            self.dict_stat_w['conv2']=np.zeros([self.conf.time_step,]+self.shape_out_conv2[1:])
            self.dict_stat_w['fc1']=np.zeros([self.conf.time_step,]+self.shape_out_fc1[1:])


        if self.conf.f_write_stat or self.conf.f_comp_act:
            self.dict_stat_w['conv1']=np.zeros([1,]+self.shape_out_conv1[1:])
            self.dict_stat_w['conv2']=np.zeros([1,]+self.shape_out_conv2[1:])
            self.dict_stat_w['fc1']=np.zeros([1,]+self.shape_out_fc1[1:])


        self.conv_p=collections.OrderedDict()
        self.conv_p['conv1_p']=np.empty(self.dict_shape['conv1_p'],dtype=np.float32)
        self.conv_p['conv2_p']=np.empty(self.dict_shape['conv2_p'],dtype=np.float32)


        #
        f_snn_training_temporal_coding = self.conf.en_train and \
                                         ((self.conf.nn_mode=='ANN' and self.conf.f_surrogate_training_model) \
                                         or (self.conf.nn_mode=='SNN' and self.conf.neural_coding=='TEMPORAL'))

        if f_snn_training_temporal_coding:
            self.init_first_spike_time = self.conf.time_fire_duration*self.conf.init_first_spike_time_n


        self.input_shape_snn = [self.conf.batch_size] + self._input_shape[1:]
        # SNN
        if self.conf.nn_mode == 'SNN' or self.conf.f_validation_snn:

            print('Neuron setup')


            print('Input shape snn: '+str(self.input_shape_snn))

            n_type = self.conf.n_type
            nc = self.conf.neural_coding

            self.list_neuron=collections.OrderedDict()

            # TODO: TF-V1
            #self.list_neuron['in'] = self.track_layer(lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf,nc,0,'in'))
            #self.list_neuron['conv1'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv1,n_type,self.fanin_conv,self.conf,nc,1,'conv1'))
            #self.list_neuron['conv2'] = self.track_layer(lib_snn.Neuron(self.shape_out_conv2,n_type,self.fanin_conv,self.conf,nc,2,'conv2'))
            #self.list_neuron['fc1'] = self.track_layer(lib_snn.Neuron(self.shape_out_fc1,'OUT',512,self.conf,nc,3,'fc1'))


            self.list_neuron['in'] = lib_snn.Neuron(self.input_shape_snn,'IN',1,self.conf,nc,0,'in')
            self.list_neuron['conv1'] = lib_snn.Neuron(self.shape_out_conv1,n_type,self.fanin_conv,self.conf,nc,1,'conv1')
            self.list_neuron['conv2'] = lib_snn.Neuron(self.shape_out_conv2,n_type,self.fanin_conv,self.conf,nc,2,'conv2')
            self.list_neuron['fc1'] = lib_snn.Neuron(self.shape_out_fc1,'OUT',512,self.conf,nc,3,'fc1')

            # modify later
            self.n_in = self.list_neuron['in']
            self.n_conv1 = self.list_neuron['conv1']
            self.n_conv2 = self.list_neuron['conv2']
            self.n_fc1 = self.list_neuron['fc1']



            self.snn_output_layer = self.n_fc1

            self.snn_output = tf.Variable(initial_value=tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_fc1.dim)),dtype=tf.float32,trainable=False)

            #
            if self.conf.f_train_time_const:
                self.dnn_act_list=collections.OrderedDict()

            #
            # TODO: make function for this code
            if self.conf.neural_coding=='TEMPORAL' and self.conf.f_load_time_const:
            #if self.conf.neural_coding=='TEMPORAL':

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

        # TODO: It should be conditional execution in SNN mode - check the condition (surrogate trained model?)
        #
        # SNN trianing - temporal coding (TTFS)
        #
        #if self.conf.neural_coding=='TEMPORAL' and self.conf.en_train:
        #if True:
        if self.conf.f_surrogate_training_model:

            #
            self.list_tk=OrderedDict()

            init_tc=self.conf.tc
            #init_td=0.0
            init_act_target_range=1.0
            init_td=init_tc*np.log(init_act_target_range)

            init_act_target_range_in=1.0
            init_td_in=init_tc*np.log(init_act_target_range_in)

            # TODO: removed
            init_ta=10.0


            # TODO: removed
#            self.list_tc=OrderedDict()
#            self.list_td=OrderedDict()
#            self.list_ta=OrderedDict()

#            self.list_tc['in'] = tf.Variable(initial_value=tf.constant(init_tc,shape=self.dict_shape_one_batch['in'],dtype=tf.float32),name="tc_in",trainable=True)
#            self.list_tc['conv1'] = tf.Variable(initial_value=tf.constant(init_tc,shape=self.dict_shape_one_batch['conv1'],dtype=tf.float32),name="tc_conv1",trainable=True)
#            self.list_tc['conv2'] = tf.Variable(initial_value=tf.constant(init_tc,shape=self.dict_shape_one_batch['conv2'],dtype=tf.float32),name="tc_conv2",trainable=True)
#            self.list_tc['fc1'] = tf.Variable(initial_value=tf.constant(init_tc,shape=self.dict_shape_one_batch['fc1'],dtype=tf.float32),name="tc_fc1",trainable=True)
#
#            self.list_td['in'] = tf.Variable(initial_value=tf.constant(init_td,shape=self.dict_shape_one_batch['in'],dtype=tf.float32),name="td_in",trainable=True)
#            self.list_td['conv1'] = tf.Variable(initial_value=tf.constant(init_td,shape=self.dict_shape_one_batch['conv1'],dtype=tf.float32),name="td_conv1",trainable=True)
#            self.list_td['conv2'] = tf.Variable(initial_value=tf.constant(init_td,shape=self.dict_shape_one_batch['conv2'],dtype=tf.float32),name="td_conv2",trainable=True)
#            self.list_td['fc1'] = tf.Variable(initial_value=tf.constant(init_td,shape=self.dict_shape_one_batch['fc1'],dtype=tf.float32),name="td_fc1",trainable=True)
#
#            self.list_ta['in'] = tf.Variable(initial_value=tf.constant(init_a,shape=self.dict_shape_one_batch['in'],dtype=tf.float32),name="ta_in",trainable=True)
#            self.list_ta['conv1'] = tf.Variable(initial_value=tf.constant(init_a,shape=self.dict_shape_one_batch['conv1'],dtype=tf.float32),name="ta_conv1",trainable=True)
#            self.list_ta['conv2'] = tf.Variable(initial_value=tf.constant(init_a,shape=self.dict_shape_one_batch['conv2'],dtype=tf.float32),name="ta_conv2",trainable=True)
#            self.list_ta['fc1'] = tf.Variable(initial_value=tf.constant(init_a,shape=self.dict_shape_one_batch['fc1'],dtype=tf.float32),name="ta_fc1",trainable=True)


            # neuron-wise
#            self.list_tk['in'] = self.track_layer(lib_snn.Temporal_kernel(
#                self.input_shape_snn, self.input_shape_snn, init_tc, init_td, init_ta,self.conf.time_window))
#            self.list_tk['conv1'] = self.track_layer(lib_snn.Temporal_kernel(
#                self.dict_shape['conv1'], self.dict_shape['conv1'], init_tc, init_td, init_ta,self.conf.time_window))
#            self.list_tk['conv2'] = self.track_layer(lib_snn.Temporal_kernel(
#                self.dict_shape['conv2'], self.dict_shape['conv2'], init_tc, init_td, init_ta,self.conf.time_window))
#            self.list_tk['fc1'] = self.track_layer(lib_snn.Temporal_kernel(
#                self.dict_shape['fc1'], self.dict_shape['fc1'], init_tc, init_td, init_ta,self.conf.time_window))

            # neuron-wise
            #s = self.dict_shape['conv2_p']
            #shape_fc1_flat = [s[0],s[1]*s[2]*s[3]]

            #self.list_tk['in'] = self.track_layer(lib_snn.Temporal_kernel(
            #    self.input_shape_snn, self.dict_shape['conv1'], init_tc, init_td, init_ta,self.conf.time_window))
            #self.list_tk['conv1'] = self.track_layer(lib_snn.Temporal_kernel(
            #    self.dict_shape['conv1'], self.dict_shape['conv2'], init_tc, init_td, init_ta,self.conf.time_window))
            #self.list_tk['conv2'] = self.track_layer(lib_snn.Temporal_kernel(
            #    self.dict_shape['conv2'], shape_fc1_flat, init_tc, init_td, init_ta,self.conf.time_window))
            #self.list_tk['fc1'] = self.track_layer(lib_snn.Temporal_kernel(
            #    self.dict_shape['fc1'], self.dict_shape['fc1'], init_tc, init_td, init_ta,self.conf.time_window))


            # layer-wise para

            # TODO: TF-V1
#            self.list_tk['in'] = self.track_layer(lib_snn.Temporal_kernel(
#                [], [], init_tc, init_td_in, init_ta,self.conf.time_window))
#            self.list_tk['conv1'] = self.track_layer(lib_snn.Temporal_kernel(
#                [], [], init_tc, init_td, init_ta,self.conf.time_window))
#            self.list_tk['conv2'] = self.track_layer(lib_snn.Temporal_kernel(
#                [], [], init_tc, init_td, init_ta,self.conf.time_window))
#            self.list_tk['fc1'] = self.track_layer(lib_snn.Temporal_kernel(
#                [], [], init_tc, init_td, init_ta,self.conf.time_window))


            self.list_tk['in'] = lib_snn.Temporal_kernel(
                [], [], init_tc, init_td_in, init_ta,self.conf.time_window)
            self.list_tk['conv1'] = lib_snn.Temporal_kernel(
                [], [], init_tc, init_td, init_ta,self.conf.time_window)
            self.list_tk['conv2'] = lib_snn.Temporal_kernel(
                [], [], init_tc, init_td, init_ta,self.conf.time_window)
            self.list_tk['fc1'] = lib_snn.Temporal_kernel(
                [], [], init_tc, init_td, init_ta,self.conf.time_window)


        # load model configuration

        # parameters load TF-V2, checkpoint version
        if self.conf.f_surrogate_training_model:
        #if False:
            self.load_layer_ann_checkpoint = tf.train.Checkpoint(conv1=self.conv1, conv2=self.conv2, fc1=self.fc1, conv1_bn=self.conv1_bn, conv2_bn=self.conv2_bn, list_tk=self.list_tk)
        else:
            self.load_layer_ann_checkpoint = tf.train.Checkpoint(conv1=self.conv1, conv2=self.conv2, fc1=self.fc1, conv1_bn=self.conv1_bn, conv2_bn=self.conv2_bn)


        # for checkpoint old version TF V1
        #self.load_layer_ann_checkpoint = tf.train.Checkpoint(conv2d=self.conv1, conv2d_1=self.conv2, dense=self.fc1, batch_normalization=self.conv1_bn, conv2_bn=self.conv2_bn)

        # TODO: TF-V1
        #pooling_type= {
        #    'max': self.track_layer(tf.layers.MaxPooling2D((2,2),(2,2),padding='SAME',data_format=data_format)),
        #    'avg': self.track_layer(tf.layers.AveragePooling2D((2,2),(2,2),padding='SAME',data_format=data_format))
        #}


#
        self.cmap=matplotlib.cm.get_cmap('viridis')
        #self.normalize=matplotlib.colors.Normalize(vmin=min(self.n_fc1.vmem),vmax=max(self.n_fc1.vmem))

    ###########################################################################
    ## processing
    ###########################################################################

    def reset_per_run_snn(self):
        self.total_spike_count=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])
        self.total_spike_count_int=np.zeros([self.num_accuracy_time_point,len(self.layer_name)+1])


    def reset_per_sample_snn(self):
        self.reset_neuron()
        #self.snn_output = np.zeros((self.num_accuracy_time_point,)+self.n_fc1.get_spike_count().numpy().shape)
        self.snn_output.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.n_fc1.dim)))
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


        # graph extract
        if False:
        #if True:
            frozen_func = self.call_ann.get_concrete_function(inputs,f_training)
            frozen_func.graph.as_graph_def()
            tf.io.write_graph(frozen_func.graph.as_graph_def(),'./tmp_graph',self.conf.model_name+'_graph.pbtxt')
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.train.export_meta_graph(filename='./tmp_graph/'+self.conf.model_name+'.meta',graph=frozen_func.graph, as_text=True)
            #tf.compat.v1.enable_eager_execution()
            assert(False)



    def preproc_snn(self,inputs,f_training):
        # reset for sample
        self.reset_per_sample_snn()

        if self.f_done_preproc == False:
            self.f_done_preproc = True
            #self.print_model_conf()
            self.reset_per_run_snn()
            self.preproc_ann_to_snn()

            if self.conf.f_surrogate_training_model:
                self.load_temporal_kernel_para()

        if self.conf.f_comp_act:
            self.save_ann_act(inputs,f_training)

        # gradient-based optimization of TC and td in temporal coding (TTFS)
        if (self.conf.neural_coding=="TEMPORAL" and self.conf.f_train_time_const):
            self.call_ann(inputs,f_training)

    def preproc_ann(self, inputs, f_training):
        if self.f_done_preproc == False:
            self.f_done_preproc=True
            #self.print_model_conf()
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

        #if self.conf.f_comp_act:
        #    self.load_act_after_w_norm()

        #self.print_act_after_w_norm()


    #
    # TODO: input neuron ?
    def load_temporal_kernel_para(self):
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



    def postproc(self, f_val_snn):

        if self.conf.nn_mode=='SNN':
            # gradient-based optimization of TC and td in temporal coding (TTFS)
            if (self.conf.neural_coding=="TEMPORAL" and self.conf.f_train_time_const):
                self.train_time_const()


        #if self.conf.nn_mode=='ANN':
        #    if f_val_snn:
        #        self.defused_bn()




    ###########################################################
    ## call function
    ###########################################################
    def call(self, inputs, f_training, f_val_snn=False):

        nn_mode = {
            'ANN': self.call_ann if not self.conf.f_surrogate_training_model else self.call_ann_surrogate_training,
            'SNN': self.call_snn if not f_training else self.call_snn_training
        }

        if self.f_1st_iter==False:
            # preprocessing
            self.preproc(inputs,f_training, f_val_snn)

            #
            if f_val_snn:
                ret_val = self.call_snn(inputs,f_training)
            else:
                ret_val = nn_mode[self.conf.nn_mode](inputs,f_training)


            # tmp
            #gr=tf.autograph.to_graph(self.call_ann(inputs,f_training).ref())
            #print(gr)
            #g = tf.Graph()
            #with g.as_default():
                #ret_val = self.call_ann(inputs,f_training)


            # post processing
            self.postproc(f_val_snn)

        else:
            self.f_skip_bn=False

            # dummy run for the eager mode in TF v1
            if self.conf.nn_mode=='SNN' and self.conf.f_surrogate_training_model:
                ret_val = self.call_ann_surrogate_training(inputs,f_training)

            # validation on SNN
            if self.conf.en_train and self.conf.f_validation_snn:
                ret_val = self.call_snn(inputs,f_training)

            ret_val = nn_mode[self.conf.nn_mode](inputs,f_training)

            self.print_model_conf()

            #self.f_1st_iter=False
        return ret_val

    #
    #@tf.function -> make the function graph?
    def call_ann(self,inputs,f_training):

        x = tf.reshape(inputs,self._input_shape)
        s_conv1 = self.conv1(x)

        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.conv1_bn(s_conv1,training=f_training)
        a_conv1 = tf.nn.relu(s_conv1_bn)
        p_conv1 = self.pool2d(a_conv1)

        #a_conv1_np = a_conv1.numpy()
        #~plt.hist(tf.reshape(a_conv1,[-1]))
        #plt.show()
        #print(tf.reduce_max(a_conv1))
        #print(tf.reshape(a_conv1,[-1]))
        #hist=tf.histogram_fixed_width(tf.reshape(a_conv1,[-1]),[0,2])
        #print(a_conv1)

        s_conv2 = self.conv2(p_conv1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.conv2_bn(s_conv2,training=f_training)
        a_conv2 = tf.nn.relu(s_conv2_bn)
        p_conv2 = self.pool2d(a_conv2)

        s_flat = tf.compat.v1.layers.flatten(p_conv2, data_format=self.data_format)

        if f_training:
            s_flat = self.dropout(s_flat,training=f_training)

        a_fc1 = self.fc1(s_flat)

        if self.conf.f_comp_act and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=a_conv1.numpy()
            self.dict_stat_w['conv2']=a_conv2.numpy()
            self.dict_stat_w['fc1']=a_fc1.numpy()

        a_out = a_fc1

        if self.f_1st_iter and self.conf.nn_mode=='ANN':
            print('1st iter')
            self.f_1st_iter = False
            self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)

        # write stat
        if (self.conf.f_write_stat) and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=np.append(self.dict_stat_w['conv1'],a_conv1.numpy(),axis=0)
            self.dict_stat_w['conv2']=np.append(self.dict_stat_w['conv2'],a_conv2.numpy(),axis=0)
            self.dict_stat_w['fc1']=np.append(self.dict_stat_w['fc1'],a_fc1.numpy(),axis=0)


        if not self.f_1st_iter and self.conf.f_train_time_const:
            print("training time constant for temporal coding in SNN")

            self.dnn_act_list['in'] = x
            self.dnn_act_list['conv1']   = a_conv1
            self.dnn_act_list['conv2']   = a_conv2
            self.dnn_act_list['fc1'] = a_fc1

        return a_out


    #
    def call_snn_validation(self,inputs,f_training):

        #if self.conf.f_surrogate_training_model:
        #    self.load_temporal_kernel_para()


        return self.call_snn(inputs,f_training)


   #
    def call_ann_surrogate_training(self,inputs,f_training):
        #if not self.f_1st_iter:
            #print(self.list_tk['fc1'].tc)

        x = tf.reshape(inputs,self._input_shape)

        #
        #t_in = self.temporal_encoding(x,'in')
        #x = self.temporal_decoding(t_in,'in')
        v_in = x
        t_in = self.list_tk['in'](v_in,'enc', self.epoch, f_training)

        v_in_dec= self.list_tk['in'](t_in, 'dec', self.epoch, f_training)
        x = v_in_dec
        ##x = t_in
        #x = t_in




        s_conv1 = self.conv1(x)

        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.conv1_bn(s_conv1,training=f_training)

        #
        #a_conv1 = tf.nn.relu(s_conv1_bn)
        #
        #self.list_v['conv1'] = s_conv1_bn

        #v_conv1 = self.list_tk['in'](s_conv1_bn, 'dec', self.epoch, f_training)
        v_conv1 = s_conv1_bn

        t_conv1 = self.list_tk['conv1'](v_conv1,'enc',self.epoch, f_training)

        #
        v_conv1_dec = self.list_tk['conv1'](t_conv1,'dec',self.epoch, f_training)
        a_conv1 = v_conv1_dec
        #a_conv1 = t_conv1

        #print(a_conv1)


        # debugging
        #print(t_conv1[0,10,10,0:10])
        #print(a_conv1[0,10,10,0:10])

        #print(tf.reduce_mean(self.conv1.kernel))



        #if False:
        #    addr=0,10,10,0
        #    print("ori: {}, enc: {}, dec: {}".format(s_conv1_bn[addr].numpy(),t_conv1[addr].numpy(),a_conv1[addr].numpy()))

        p_conv1 = self.pool2d(a_conv1)



        s_conv2 = self.conv2(p_conv1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.conv2_bn(s_conv2,training=f_training)

        #
        #a_conv2 = tf.nn.relu(s_conv2_bn)
        #
        #self.list_v['conv2'] = s_conv2_bn


        #v_conv2 = self.list_tk['conv1'](s_conv2_bn,'dec',self.epoch, f_training)
        v_conv2 = s_conv2_bn

        t_conv2 = self.list_tk['conv2'](v_conv2,'enc',self.epoch, f_training)

        #
        v_conv2_dec = self.list_tk['conv2'](t_conv2,'dec',self.epoch, f_training)
        a_conv2 = v_conv2_dec
        #a_conv2 = t_conv2

        p_conv2 = self.pool2d(a_conv2)


        s_flat = tf.compat.v1.layers.flatten(p_conv2, data_format=self.data_format)

        if f_training:
            s_flat = self.dropout(s_flat,training=f_training)

        #

        #v_fc1 = self.list_tk['conv2'](s_flat,'dec',self.epoch, f_training)
        v_fc1 = s_flat

        a_fc1 = self.fc1(v_fc1)
        #
        #self.list_v['fc1'] = a_fc1


        #v_fc1 = a_fc1
        #t_fc1 = self.list_tk['fc1'](v_fc1,'enc',self.epoch, f_training)
        #v_fc1 = self.list_tk['fc1'](t_fc1,'dec',self.epoch, f_training)
        #a_fc1 = v_fc1

        #print(a_fc1)


        a_out = a_fc1
        #a_out = t_fc1

        #print(a_out)




        #
        #if self.f_1st_iter and self.conf.nn_mode=='ANN':
        if self.f_1st_iter:
            print('1st iter')
            self.f_1st_iter = False
            #self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)

        #
        if not self.f_1st_iter and self.conf.f_train_time_const:
            print("training time constant for temporal coding in SNN")

            self.dnn_act_list['in'] = x
            self.dnn_act_list['conv1']   = a_conv1
            self.dnn_act_list['conv2']   = a_conv2
            self.dnn_act_list['fc1'] = a_fc1

        if self.conf.f_comp_act and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=a_conv1.numpy()
            self.dict_stat_w['conv2']=a_conv2.numpy()
            self.dict_stat_w['fc1']=a_fc1.numpy()

        #print(self.list_tk['in'].tc)
        #print(self.list_tk['in'].td)
        #print(self.list_tk['conv1'].tc)
        #print(self.list_tk['conv1'].td)
        #print(self.list_tk['conv2'].tc)
        #print(self.list_tk['conv2'].td)


        # debugging

        if False:
        #if True:
            x = v_in_dec
            #x = v_conv1
            print(tf.reduce_mean(x))
            plt.hist(tf.reshape(x,[-1]))
            plt.show()


        if False:
        #if True:
            print('surrogate model')
            print(tf.reduce_mean(self.conv1.kernel))
            print(tf.reduce_mean(self.conv1.bias))
            print(tf.reduce_mean(self.conv2.kernel))
            print(tf.reduce_mean(self.conv2.bias))
            print(tf.reduce_mean(self.fc1.kernel))
            print(tf.reduce_mean(self.fc1.bias))
            print(self.conv1_bn.trainable_variables)
            print(self.conv2_bn.trainable_variables)



        return a_out




    #
    def call_ann_surrogate_training_one_para_per_layer(self,inputs,f_training):
        #if not self.f_1st_iter:
            #print(self.list_tk['fc1'].tc)

        x = tf.reshape(inputs,self._input_shape)

        #
        #t_in = self.temporal_encoding(x,'in')
        #x = self.temporal_decoding(t_in,'in')
        v_in = x
        t_in = self.list_tk['in'](v_in,'enc', self.epoch, f_training)
        v_in_dec= self.list_tk['in'](t_in, 'dec', self.epoch, f_training)
        x = v_in_dec
        #x = t_in


        #print(x)

        s_conv1 = self.conv1(x)

        # conv - deconv test
        #s_conv1_tr = self.conv1_tr(x)
        #s_conv1_tr_cv = self.conv1_tr_cv(s_conv1_tr)
        #
        #print(x.shape)
        #print(s_conv1_tr.shape)
        #print(s_conv1_tr_cv.shape)
        #print(s_conv1.shape)
        #
        ##print('shape')
        ##print(self.conv1_tr.kernel.shape)
        ##print(self.conv1_tr_cv.kernel.shape)
        #
        #print(tf.equal(s_conv1,s_conv1_tr_cv))
        #print(s_conv1[0,10,10,0])
        #print(s_conv1_tr_cv[0,10,10,0])
        #s_conv1 = s_conv1_tr_cv



        if self.f_skip_bn:
            s_conv1_bn = s_conv1
        else:
            s_conv1_bn = self.conv1_bn(s_conv1,training=f_training)

        #
        #a_conv1 = tf.nn.relu(s_conv1_bn)
        #
        #self.list_v['conv1'] = s_conv1_bn

        v_conv1 = s_conv1_bn
        t_conv1 = self.list_tk['conv1'](v_conv1,'enc',self.epoch, f_training)
        v_conv1_dec = self.list_tk['conv1'](t_conv1,'dec',self.epoch, f_training)
        a_conv1 = v_conv1_dec

        #print(a_conv1)

        if False:
            addr=0,10,10,0
            print("ori: {}, enc: {}, dec: {}".format(s_conv1_bn[addr].numpy(),t_conv1[addr].numpy(),a_conv1[addr].numpy()))

        p_conv1 = self.pool2d(a_conv1)



        s_conv2 = self.conv2(p_conv1)
        if self.f_skip_bn:
            s_conv2_bn = s_conv2
        else:
            s_conv2_bn = self.conv2_bn(s_conv2,training=f_training)

        #
        #a_conv2 = tf.nn.relu(s_conv2_bn)

        v_conv2 = s_conv2_bn
        t_conv2 = self.list_tk['conv2'](v_conv2,'enc',self.epoch, f_training)
        v_conv2_dec = self.list_tk['conv2'](t_conv2,'dec',self.epoch, f_training)
        a_conv2 = v_conv2_dec

        p_conv2 = self.pool2d(a_conv2)


        s_flat = tf.layers.flatten(p_conv2)

        if f_training:
            s_flat = self.dropout(s_flat,training=f_training)

        #print(s_flat)
        #
        a_fc1 = self.fc1(s_flat)
        #


        #v_fc1 = a_fc1
        #t_fc1 = self.list_tk['fc1'](v_fc1,'enc',self.epoch, f_training)
        #v_fc1 = self.list_tk['fc1'](t_fc1,'dec',self.epoch, f_training)
        #a_fc1 = v_fc1

        #print(a_fc1)


        a_out = a_fc1
        #a_out = t_fc1

        #print(a_out)




        #
        #if self.f_1st_iter and self.conf.nn_mode=='ANN':
        if self.f_1st_iter:
            print('1st iter')
            self.f_1st_iter = False
            #self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)

        #
        if not self.f_1st_iter and self.conf.f_train_time_const:
            print("training time constant for temporal coding in SNN")

            self.dnn_act_list['in'] = x
            self.dnn_act_list['conv1']   = a_conv1
            self.dnn_act_list['conv2']   = a_conv2
            self.dnn_act_list['fc1'] = a_fc1

        if self.conf.f_comp_act and (not self.f_1st_iter):
            self.dict_stat_w['conv1']=a_conv1.numpy()
            self.dict_stat_w['conv2']=a_conv2.numpy()
            self.dict_stat_w['fc1']=a_fc1.numpy()

        #print(self.list_tk['in'].tc)
        #print(self.list_tk['in'].td)
        #print(self.list_tk['conv1'].tc)
        #print(self.list_tk['conv1'].td)
        #print(self.list_tk['conv2'].tc)
        #print(self.list_tk['conv2'].td)
        return a_out


    ###########################################################
    ## SNN training - temporal coding (TTFS), surrogate
#    ###########################################################
#    # TODO: Remove
#    def temporal_encoding_kernel(self, x, l_name):
#        #td = 0.0
#        #a = 10.0
#        #tc = 20.0
#        #tw = self.conf.time_window
#
#        tc=self.list_tc[l_name]
#        td=self.list_td[l_name]
#        a=self.list_ta[l_name]
#
#        eps=1.0E-10
#
#        x=tf.nn.relu(x)
#        #x=tf.add(x,eps)
#        x=tf.divide(x,a)
#        x=tf.nn.relu(x)
#        x=tf.add(x,eps)
#
#
#        #print(x.shape)
#        #print(tf.add(x,eps).shape)
#        #print(a.shape)
#
#        t_float = tf.subtract(td,tf.multiply(tf.math.log(x),tc))
#
#        #t = tf.quantization.fake_quant_with_min_max_vars(t_float,0,tf.pow(tf.constant(2.0),tf.constant(16.0))-1,16)
#        if self.epoch > 100:
#            t = tf.quantization.fake_quant_with_min_max_vars(t_float,0,tf.pow(2.0,16.0)-1,16)
#        else :
#            t = t_float
#
#        #print(t_float)
#        #print(t)
#
#
#        #idx=np.where(tf.is_nan(t).numpy())[1:]
#        #idx=np.where(t.numpy()<3)
#        #idx=np.where(t.numpy()<3)
#
#        #print('idx')
#        #print(idx[0])
#
#        #idx_ori = np.where(t.numpy()>0.5)
#        #print(idx_ori)
#        #print(idx_ori[1])
#        #idx=[zip(np.zeros(shape=idx_ori[1].shape,dtype=np.int32),idx_ori[1],idx_ori[2],idx_ori[3])]
#        #idx=[0,idx_ori[1][0],idx_ori[2][0],idx_ori[3][0]]
#        #print(idx)
#        #print(a[idx])
#
#
#
#        #if tf.is_nan(tf.reduce_mean(t)):
#        #    #print(t)
#        #    #print(a)
#        #    #idx=[1,]+np.where(tf.is_nan(t).numpy())[1:]
#        #    #print(a[idx])
#        #    idx_ori = np.where(tf.is_nan(t).numpy())
#        #    #print(idx_ori)
#        #    #idx=[0,idx_ori[1],idx_ori[2],idx_ori[3]]
#        #    idx=[0,idx_ori[1][0],idx_ori[2][0],idx_ori[3][0]]
#        #    print(a[idx])
#        #    print('encoding')
#        #    assert(False)
#
#        return t
#
#    def temporal_encoding(self, x, l_name):
#        #td = 0.0
#        #a = 10.0
#        #tc = 20.0
#        tw = self.conf.time_window
#
#        tc=self.list_tc[l_name]
#        td=self.list_td[l_name]
#        a=self.list_ta[l_name]
#
#
#        #eps=0.0000000001
#        eps=1.0E-10
#
#
#        #x=tf.nn.relu(x)
#        #ret = tf.math.ceil(tf.subtract(td,tf.multiply(tf.divide(x,a),tc)))
#
#        # kernel
#        #t = tf.subtract(td,tf.multiply(tf.math.log(tf.divide(x+eps,a)),tc))
#        t = self.temporal_encoding_kernel(x,l_name)
#
#        # time window
#        #t=tf.math.minimum(t, tw*10)
#        #t=tf.math.minimum(t, tw*8)
#        #t=tf.math.minimum(t, tw*6)
#        #t=tf.math.minimum(t, tw*4)
#        #t=tf.math.minimum(t, tw*2)
#
#        if self.epoch > 100:
#            t=tf.math.minimum(t, tw)
#
#        #t = tf.math.minimum(tf.math.maximum(0,t),tw)
#
#        # TODO: apply time window, if T is larger than certain time window, it goes to zero
#
#        #tf.where(ret>tw,tf.constant(0.0,shape=ret.shape,dtype=tf.float32),ret)
#
#        #print(x)
#        #print(ret)
#
#        #print(tf.math.log(tf.divide(eps,a)))
#
#        #assert(False)
#
#        return t
#
#
#    def temporal_decoding(self, t, l_name):
#        #td = 0.0
#        #a = 10.0
#        #tc = 20.0
#
#        tc=self.list_tc[l_name]
#        td=self.list_td[l_name]
#        a=self.list_ta[l_name]
#
#        x = tf.multiply(a,tf.exp(tf.divide(tf.subtract(td,t),tc)))
#
#        #
#        #if tf.is_nan(tf.reduce_mean(x)):
#        #    print(x)
#        #    print('decoding')
#        #    assert(False)
#
#        return x





    #
    def call_snn(self,inputs,f_training):

        #
        plt.clf()


        #
        inputs = tf.reshape(inputs,self._input_shape)

        #
        for t in range(self.tw):
            if self.verbose == True:
                print('time: '+str(t))

            self.bias_control(t)

            #print(a_in)

            a_in = self.n_in(inputs,t)

            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)
            #a_conv1_tr = self.conv1_tr(a_conv1)
            #a_conv1_tr_cv = self.conv1_tr_cv(a_conv1_tr)
            #print(tf.equal(a_conv1,a_conv1_tr_cv))
            #print(a_conv1[0,10,10,0:10])
            #print(a_conv1_tr_cv[0,10,10,0])
            #a_conv1 = a_conv1_tr_cv



            #x=a_conv1
            #print('min: {:}, max:{:}'.format(tf.reduce_min(x),tf.reduce_max(x)))


            p_conv1 = self.max_pool(a_conv1, 'conv1')

            s_conv2 = self.conv2(p_conv1)
            a_conv2 = self.n_conv2(s_conv2,t)
            p_conv2 = self.max_pool(a_conv2, 'conv2')

            flat = tf.compat.v1.layers.flatten(p_conv2, data_format=self.data_format)

            s_fc1 = self.fc1(flat)
            a_fc1 = self.n_fc1(s_fc1,t)

            #
            if self.f_1st_iter == False:

                if False:
                #if True:
                    x = self.n_conv1.vmem
                    print(tf.reduce_mean(x))
                    plt.hist(tf.reshape(x,[-1]))
                    plt.show()

                if False:
                #if True:

                    print(self.n_conv1.vmem[0,10,10,0:10])
                    print('w')
                    print(tf.reduce_mean(self.conv1.kernel))
                    print(tf.reduce_mean(self.conv1.bias))
                    print(tf.reduce_mean(self.conv2.kernel))
                    print(tf.reduce_mean(self.conv2.bias))
                    print(tf.reduce_mean(self.fc1.kernel))
                    print(tf.reduce_mean(self.fc1.bias))
                    print(self.conv1_bn.trainable_variables)
                    print(self.conv2_bn.trainable_variables)

                    #assert False

                    # test
                    print(self.n_in.time_const_integ)
                    print(self.n_in.time_delay_integ)
                    print(self.n_in.time_const_fire)
                    print(self.n_in.time_delay_fire)
                    print('')

                    print(self.n_conv1.time_const_integ)
                    print(self.n_conv1.time_delay_integ)
                    print(self.n_conv1.time_const_fire)
                    print(self.n_conv1.time_delay_fire)
                    print('')

                    print(self.n_conv2.time_const_integ)
                    print(self.n_conv2.time_delay_integ)
                    print(self.n_conv2.time_const_fire)
                    print(self.n_conv2.time_delay_fire)
                    print('')

                    print(self.n_fc1.time_const_integ)
                    print(self.n_fc1.time_delay_integ)
                    print(self.n_fc1.time_const_fire)
                    print(self.n_fc1.time_delay_fire)
                    print('')

                #if True:
                if False:
                    print(self.list_tk['in'].tw)
                    print(self.list_tk['conv1'].tw)
                    print(self.list_tk['conv2'].tw)
                    #print(self.list_tk['fc1'].tw)



                #print(self.count_accuracy_time_point)
                if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                    #output=self.n_fc1.vmem
                    #self.recoding_ret_val(output)
                    self.recoding_ret_val()

        if self.conf.f_entropy and (not self.f_1st_iter):
            self.cal_entropy()


        #
        self.bias_restore()


        if self.f_1st_iter:
            self.f_1st_iter = False

            self.conv1_bn(s_conv1,training=f_training)
            self.conv2_bn(s_conv2,training=f_training)

            return 0

        else:
            self.get_total_residual_vmem()
            spike_zero = tf.reduce_sum(self.snn_output,axis=[0,2])

            #if np.any(num_spike_count.numpy() == 0):
            if np.any(spike_zero.numpy() == 0):
                print('spike count 0')
                #print(num_spike_count.numpy())
                #a = input("press any key to exit")
                #os.system("Pause")
                #raw_input("Press any key to exit")
                #sys.exit(0)


        #print(self.list_neuron['in'].time_const_integ)
        #print(self.list_neuron['in'].time_delay_integ)
#        print('')
#        print(self.list_neuron['in'].time_const_fire)
#        print(self.list_neuron['in'].time_delay_fire)
#
#        print(self.list_neuron['conv1'].time_const_integ)
#        print(self.list_neuron['conv1'].time_delay_integ)
#        print(self.list_neuron['conv1'].time_const_fire)
#        print(self.list_neuron['conv1'].time_delay_fire)
#
#        print(self.list_neuron['conv2'].time_const_integ)
#        print(self.list_neuron['conv2'].time_delay_integ)
#        print(self.list_neuron['conv2'].time_const_fire)
#        print(self.list_neuron['conv2'].time_delay_fire)
#
#        print(self.list_neuron['fc1'].time_const_integ)
#        print(self.list_neuron['fc1'].time_delay_integ)
        #print(self.list_neuron['fc1'].time_const_fire)
        #print(self.list_neuron['fc1'].time_delay_fire)

        #print(self.list_neuron['conv1'].time_const_fire)
        #print('time_const_check')

        # TODO: analysis
        #if False:
        if self.conf.f_comp_act and (not self.f_1st_iter):
            print('compare activation')
            print(self.list_tk['conv1'].in_enc[0,:,1,2])
            print(self.list_tk['conv1'].out_enc[0,:,1,2])
            print(self.list_tk['conv1'].out_dec[0,:,1,2])
            print(self.dict_stat_w['conv1'][0,:,1,2])
            print(self.list_layer['conv1'].bias)
            print(self.list_neuron['conv1'].vmem[0,:,1,2])
            #print(self.list_neuron['conv1'].spike_counter[0,:,1,2])
            print(self.list_neuron['conv1'].first_spike_time[0,:,1,2])
            #self.dict_stat_w['conv1']=a_conv1.numpy()
            #self.dict_stat_w['conv2']=a_conv2.numpy()
            #self.dict_stat_w['fc1']=a_fc1.numpy()




        #plt.hist(self.n_conv1.vmem.numpy().flatten())
        #plt.show()
        #return spike_count
        return self.snn_output



    #
    def call_snn_training(self,inputs,f_training):

        snn_training_sel = {
            'TEMPORAL': self.call_snn_training_temporal
        }
        snn_training_func=snn_training_sel[self.conf.neural_coding](inputs, True)


    ###########################################################
    ## SNN training - temporal coding (TTFS)
    ###########################################################

    def call_snn_training_temporal(self, inputs, f_training):
        #
        plt.clf()

        #
        inputs = tf.reshape(inputs,self._input_shape)

        #
        for t in range(self.tw):
            if self.verbose == True:
                print('time: '+str(t))

            self.bias_control(t)


            a_in = self.n_in(inputs,t)

            s_conv1 = self.conv1(a_in)
            a_conv1 = self.n_conv1(s_conv1,t)
            #a_conv1_tr = self.conv1_tr(a_conv1)
            #a_conv1_tr_cv = self.conv1_tr_cv(a_conv1_tr)


            p_conv1 = self.max_pool(a_conv1, 'conv1')

            s_conv2 = self.conv2(p_conv1)
            a_conv2 = self.n_conv2(s_conv2,t)
            p_conv2 = self.max_pool(a_conv2, 'conv2')

            flat = tf.layers.flatten(p_conv2)

            s_fc1 = self.fc1(flat)
            a_fc1 = self.n_fc1(s_fc1,t)

            #
            if self.f_1st_iter == False:
                if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
                    #output=self.n_fc1.vmem
                    #self.recoding_ret_val(output)
                    self.recoding_ret_val()

        if self.conf.f_entropy and (not self.f_1st_iter):
            self.cal_entropy()


        if self.f_1st_iter:
            self.f_1st_iter = False

            self.conv1_bn(s_conv1,training=f_training)
            self.conv2_bn(s_conv2,training=f_training)

            return 0

        else:
            self.get_total_residual_vmem()
            spike_zero = tf.reduce_sum(self.snn_output,axis=[0,2])

            #if np.any(num_spike_count.numpy() == 0):
            if np.any(spike_zero.numpy() == 0):
                print('spike count 0')
                #print(num_spike_count.numpy())
                #a = input("press any key to exit")
                #os.system("Pause")
                #raw_input("Press any key to exit")
                #sys.exit(0)




        #plt.hist(self.n_conv1.vmem.numpy().flatten())
        #plt.show()
        #return spike_count
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







    ###########################################################################
    ## weignt normalization
    ###########################################################################

    def w_norm_layer_wise(self):

        f_norm=np.max
        #f_norm=np.mean

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

        deep_layer_const = 1.0

        self.conv1.kernel = self.conv1.kernel/self.norm['conv1']*deep_layer_const
        self.conv1.bias = self.conv1.bias/self.norm_b['conv1']

        self.conv2.kernel = self.conv2.kernel/self.norm['conv2']*deep_layer_const
        self.conv2.bias = self.conv2.bias/self.norm_b['conv2']


        self.fc1.kernel = self.fc1.kernel/self.norm['fc1']*deep_layer_const
        self.fc1.bias = self.fc1.bias/self.norm_b['fc1']

    #
    def data_based_w_norm(self):
        #stat_file='./stat/dist_act_trainset_'+self.conf.model_name
        #stat_file='./stat/dist_act_neuron_trainset_'+self.conf.model_name
        #stat_file='./stat/act_n_trainset_test_'+self.conf.model_name

        path_stat= '../stat/'
        f_name_stat='act_n_train'
        stat_conf=['max','mean','max_999','max_99','max_98']
        f_stat=collections.OrderedDict()
        r_stat=collections.OrderedDict()

        # choose one
        #stat='max'
        #stat='mean'
        stat='max_999'
        #stat='max_99'
        #stat='max_98'

        for idx_l, l in enumerate(self.layer_name):
            key=l+'_'+stat

            f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])

            #if self.conf.f_ws:
                #self.dict_stat_r['conv1']=self.dict_stat_r['conv1']/(1-1/np.power(2,8))

            #print(np.shape(self.dict_stat_r[l]))

        self.w_norm_layer_wise()


    #
    def load_act_after_w_norm(self):
        path_stat= '../stat/'
        f_name_stat='act_n_train_after_w_norm_max_999'
        f_stat=collections.OrderedDict()
        r_stat=collections.OrderedDict()

        # choose one
        #stat='max'
        stat='mean'
        #stat='min'
        #stat='max_75'
        #stat='max_25'

        for idx_l, l in enumerate(self.layer_name):
            key=l+'_'+stat

            f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'r')
            r_stat[key]=csv.reader(f_stat[key])

            for row in r_stat[key]:
                self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.dict_shape[l][1:])


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

    def _model_based_norm(self, layer, axis):
        w_in_sum = tf.reduce_sum(tf.maximum(layer.kernel,0),axis=axis)
        w_in_max = tf.reduce_max(w_in_sum)
        layer.kernel = layer.kernel/w_in_max

    def model_based_norm(self):
        print('model based_norm')
        w_in_sum_conv1 = tf.reduce_sum(tf.maximum(self.conv1.kernel,0),axis=[0,1,2])
        #w_in_max_conv1 = tf.reduce_max(w_in_sum_conv1+self.conv1.bias)
        w_in_max_conv1 = tf.reduce_max(w_in_sum_conv1)
        self.conv1.kernel = self.conv1.kernel/w_in_max_conv1

        w_in_sum_conv2 = tf.reduce_sum(tf.maximum(self.conv2.kernel,0),axis=[0,1,2])
        #w_in_max_conv2 = tf.reduce_max(w_in_sum_conv2+self.conv2.bias)
        w_in_max_conv2 = tf.reduce_max(w_in_sum_conv2)
        self.conv2.kernel = self.conv2.kernel/w_in_max_conv2

        w_in_sum_fc1 = tf.reduce_sum(tf.maximum(self.fc1.kernel,0),axis=[0])
        #w_in_max_fc1 = tf.reduce_max(w_in_sum_fc1+self.fc1.bias)
        w_in_max_fc1 = tf.reduce_max(w_in_sum_fc1)
        # w_in_min_fc1 = tf.reduce_min(w_in_sum_fc1)
        #self.fc1.kernel = self.fc1.kernel-kernel_min/2
        self.fc1.kernel = self.fc1.kernel/w_in_max_fc1

        if self.verbose == True:
            print('w_in_max_conv1:' +str(w_in_max_conv1))
            print('w_in_max_conv2:' +str(w_in_max_conv2))
            print('w_in_max_conv2_1:' +str(w_in_max_conv2_1))
            print('w_in_max_fc1:' +str(w_in_max_fc1))


    ###########################################################################
    ## misc.
    ###########################################################################

    def max_pool(self, activation, layer_name):
        neuron = self.list_neuron[layer_name]
        pool_name = layer_name+'_p'

        spike_count=neuron.get_spike_count()
        shape=self.dict_shape[pool_name]

        if self.conf.f_spike_max_pool:
            pool = lib_snn.spike_max_pool(activation,spike_count,shape)
        else:
            pool = self.pool2d(activation)

        return pool

    def fused_bn(self):
        if self.conf.verbose:
            print('fused_bn')

        self.conv_bn_fused(self.conv1, self.conv1_bn, 1.0)
        self.conv_bn_fused(self.conv2, self.conv2_bn, 1.0)

    #
    def defused_bn(self):
        if self.conf.verbose:
            print('defused_bn')

        self.conv_bn_defused(self.conv1, self.conv1_bn)
        self.conv_bn_defused(self.conv2, self.conv2_bn)

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

        #print(gamma)
        #print(beta)

        #return kernel, bias


    #
    def conv_bn_defused(self, conv, bn):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        conv.kernel = conv.kernel/inv
        conv.bias = (conv.bias-beta)/inv+mean



    def fc_bn_fused(self, conv, bn, time_step):
        gamma=bn.gamma
        beta=bn.beta
        mean=bn.moving_mean
        var=bn.moving_variance
        ep=bn.epsilon
        inv=math_ops.rsqrt(var+ep)
        inv*=gamma

        #conv.kernel = conv.kernel*gamma/var_e/time_step
        conv.kernel = conv.kernel*math_ops.cast(inv,conv.kernel.dtype)
        conv.bias = ((conv.bias-mean)*inv+beta)

        #print(gamma)
        #print(beta)

        #return kernel, bias


    def print_model_conf(self):
        # print model configuration
        print('Input   N: '+str(self.in_shape))

        print('Conv1   S: '+str(self.conv1.kernel.get_shape()))
        print('Conv1   N: '+str(self.shape_out_conv1))
        print('Pool1   N: '+str(self.shape_out_conv1_p))

        print('Conv2   S: '+str(self.conv2.kernel.get_shape()))
        print('Conv2   N: '+str(self.shape_out_conv2))
        print('Pool2   N: '+str(self.shape_out_conv2_p))

        print('fc1     S: '+str(self.fc1.kernel.get_shape()))
        print('fc1     N: '+str(self.shape_out_fc1))


    def print_act_d(self):
        print('print activation')

        fig, axs = plt.subplots(6,3)

        axs=axs.ravel()

        #for idx_l, (name_l,stat_l) in enumerate(self.dict_stat_r):
        for idx, (key, value) in enumerate(self.dict_stat_r.items()):
            axs[idx].hist(value.flatten())

        plt.show()


    def plot(self, x, y, mark):
        #plt.ion()
        #plt.hist(self.n_fc1.vmem)
        plt.plot(x, y, mark)
        plt.draw()
        plt.pause(0.00000001)
        #plt.ioff()

    def scatter(self, x, y, color, marker='o'):
        #plt.ion()
        plt.scatter(x, y, c=color, s=1, marker=marker)
        plt.draw()
        plt.pause(0.00000001)
        #plt.ioff()


    ###########################################################################
    ## Analysis
    ###########################################################################

    def get_total_residual_vmem(self):
        len=self.total_residual_vmem.shape[0]
        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            idx=idx_n-1
            if nn!='in' or nn!='fc1':
                self.total_residual_vmem[idx]+=tf.reduce_sum(tf.abs(n.vmem))
                self.total_residual_vmem[len-1]+=self.total_residual_vmem[idx]

    def get_total_isi(self):
        isi_count=np.zeros(self.conf.time_step)

        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc1':
                isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
                isi_count_n.resize(self.conf.time_step)
                isi_count = isi_count + isi_count_n

        return isi_count

    def get_total_spike_amp(self):
        spike_amp=np.zeros(self.spike_amp_kind)
        #print(range(0,self.spike_amp_kind)[::-1])
        #print(np.power(0.5,range(0,self.spike_amp_kind)))

        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            if nn!='in' or nn!='fc1':
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

        for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
            idx=idx_n-1
            if nn!='in' and nn!=self.layer_name[-1]:
                spike_count_int[idx]=tf.reduce_sum(n.get_spike_count_int())
                spike_count_int[len-1]+=spike_count_int[idx]
                spike_count[idx]=tf.reduce_sum(n.get_spike_count())
                spike_count[len-1]+=spike_count[idx]

                #print(spike_count_int[idx])

        return [spike_count_int, spike_count]


    def comp_act_rate(self,t):
        self.total_comp_act[t,-1]=0.0
        for idx_l, l in enumerate(self.layer_name):
            if l !='fc1':
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/(float)(t+1)-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]

    def comp_act_ws(self,t):
        self.total_comp_act[t,-1]=0.0
        for idx_l, l in enumerate(self.layer_name):
            if l !='fc1':
                #self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,idx_l]=np.mean(np.abs(self.list_neuron[l].spike_counter.numpy().flatten()/((float)(t+1)/(float)(self.conf.p_ws))-self.dict_stat_w[l].flatten()))
                self.total_comp_act[t,-1]+=self.total_comp_act[t,idx_l]

    def comp_act_pro(self,t):
        self.comp_act_ws(t)

    def save_ann_act(self,inputs,f_training):
        if self.conf.f_surrogate_training_model:
            self.call_ann_surrogate_training(inputs,f_training)
        else:
            self.call_ann(inputs,f_training)


    #
    def cal_entropy(self):
        length = 4

        for idx_l, l in enumerate(self.layer_name):
            if l !='fc1':
                #print(self.dict_stat_w[l].shape)
                self.dict_stat_w[l][np.nonzero(self.dict_stat_w[l])]=1.0

                #print(self.dict_stat_w[l])

                #print(np.array2string(str(self.dict_stat_w[l]),max_line_width=4))
                #print(self.dict_stat_w[l].shape)

                num_words = self.dict_stat_w[l].shape[0]/length
                tmp = np.zeros((num_words,)+(self.dict_stat_w[l].shape[1:]))

                for idx in range(num_words):
                    for idx_length in range(length):
                        tmp[idx] += self.dict_stat_w[l][idx*length+idx_length]*np.power(2,idx_length)

                #print(tmp)

                #plt.hist(tmp.flatten())
                #plt.show()
                #print(tmp.shape)

                print(stats.entropy(np.histogram(tmp.flatten())))


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
                if t==0:
                    self.bias_enable()
                else:
                    if tf.equal(tf.reduce_max(a_in),0.0):
                        self.bias_enable()
                    else:
                        self.bias_disable()


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



    ###########################################################
    ## training time constant for temporal coding (TTFS)
    ###########################################################

    def train_time_const(self):

        print("models: train_time_const")

        # train_time_const
        name_layer_prev=''
        for name_layer, layer in self.list_neuron.items():
            if not ('fc1' in name_layer):
                dnn_act = self.dnn_act_list[name_layer]
                self.list_neuron[name_layer].train_time_const_fire(dnn_act)

            if not ('in' in name_layer):
                self.list_neuron[name_layer].set_time_const_integ(self.list_neuron[name_layer_prev].time_const_fire)

            name_layer_prev = name_layer


        # train_time_delay
        name_layer_prev=''
        for name_layer, layer in self.list_neuron.items():
            if not ('fc1' in name_layer or 'in' in name_layer):
                dnn_act = self.dnn_act_list[name_layer]
                self.list_neuron[name_layer].train_time_delay_fire(dnn_act)

            if not ('in' in name_layer or 'conv1' in name_layer):
                self.list_neuron[name_layer].set_time_delay_integ(self.list_neuron[name_layer_prev].time_delay_fire)

            name_layer_prev = name_layer


#        if self.conf.f_tc_based:
#            # update integ and fire time
#            name_layer_prev=''
#            for name_layer, layer in self.list_neuron.items():
#                if not ('fc3' in name_layer):
#                    self.list_neuron[name_layer].set_time_fire(self.list_neuron[name_layer].time_const_fire*self.conf.n_tau_fire_start)
#
#                if not ('in' in name_layer):
#                    self.list_neuron[name_layer].set_time_integ(self.list_neuron[name_layer_prev].time_const_integ*self.conf.n_tau_fire_start)
#
#                name_layer_prev = name_layer


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












