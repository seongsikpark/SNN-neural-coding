import tensorflow as tf
import tensorflow.contrib.eager as tfe

import tensorflow.keras.layers as layers

from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers

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

from scipy import stats
from scipy import sparse

from operator import itemgetter

#
class mlp_mnist(tf.keras.Model):
    def __init__(self, data_format, conf):
        super(mlp_mnist, self).__init__(name='')

        # configuration
        self.conf = conf

        #
        self.dim_i = 784
        self.dim_h = 800
        self.dim_o = conf.num_class

        self._input_shape = [-1,self.dim_i]
        self._hidden_shape = [-1,self.dim_h]
        self._output_shape = [-1,self.dim_o]

        self.i_shape = (self.conf.batch_size,) + tuple(self._input_shape[1:])
        self.h_shape = (self.conf.batch_size,) + tuple(self._hidden_shape[1:])
        self.o_shape = (self.conf.batch_size,) + tuple(self._output_shape[1:])

        # internal
        self.f_model_load_done = False
        self.f_1st_iter = True

        self.type_call = {
            'ANN': self.call_ann,
            'SNN': self.call_snn
        }

        #
        regularizer_type = {
            'L1': regularizers.l1_regularizer(conf.lamb),
            'L2': regularizers.l2_regularizer(conf.lamb)
        }

        kernel_regularizer = regularizer_type[conf.regularizer]
        kernel_initializer = initializers.xavier_initializer(True)

        #
        self.list_l = []        # list - layers
        self.list_s = []        # list - PSP
        self.list_a = []        # list - activation values
        self.list_n = []        # list - neurons
        self.list_tpsp = []     # list - total psp

        self.list_a.append([])

        self.list_l.append(layers.Dense(
            self.dim_h,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        ))

        self.lists.append([])
        self.list_n.append(tf.nn.relu)
        self.list_a.append([])


        self.list_l.append(layers.Dense(
            self.dim_o,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        ))


        self.list_s.append([])
        self.list_n.append(tf.nn.relu)
        self.list_a.append([])

        # neurons
        if self.conf.nn_mode == 'SNN':
            print('SNN mode')
            self.list_n=[]


            nc = self.conf.neural_coding

            self.list_n.append(lib_snn.Neuron(
                self.i_shape,
                'IN',
                1,
                self.conf,
                nc
            ))

            self.list_n.append(lib_snn.Neuron(
                self.h_shape,
                'LIF',
                1,
                self.conf,
                nc
            ))

            self.list_n.append(lib_snn.Neuron(
                self.o_shape,
                'LIF',
                1,
                self.conf,
                nc
            ))

            # total psp


            self.accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
            self.accuracy_time_point.append(conf.time_step)
            self.num_accuracy_time_point = len(self.accuracy_time_point)
            self.count_accuracy_time_point = 0

            self.spike_count = np.zeros((self.num_accuracy_time_point,)+self.o_shape)
            #self.spike_count = np.zeros(self.o_shape)
            #self.spike_count = np.zeros([self.num_accuracy_time_point,].extend(self.o_shape))

    def add_variables(self):
        for _, l in enumerate(self.list_l):
            self.list_tpsp.append(np.zeros(l.kernel.numpy().shape))

    def reset_variables(self):
        # tpsp
        for idx_l, l in enumerate(self.list_tpsp):
            self.list_tpsp[idx_l] = np.zeros(l.shape)

    def preproc(self):
        print('not define yet')
        os._exit(0)

    def batch_padding(self,input_tensor):
        d = input_tensor.numpy().shape[0]
        t = self.i_shape[0]
        g = t-d

        return tf.concat([input_tensor,tf.zeros((g,)+self.i_shape[1:])],axis=0)

    def total_spike_count_int(self):
        print('total_spike_count_int: not implemented yet')

    def total_spike_count(self):
        print('total_spike_count: not implemented yet')


    def call(self, input_tensor, f_training):
        if self.f_model_load_done == True:
            if self.f_1st_iter == True:
                self.add_variables()
                #    self.preproc()
                self.f_1st_iter = False

            # modify later
            #if self.conf.f_fused_bn == True:
            #    ret_val = self.type_call_fused_bn[self.conf.nn_mode](input_tensor, f_training)
            #else:
            #    ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training)

            if self.conf.nn_mode=='SNN':
                if input_tensor.numpy().shape != self.i_shape:
                    input_tensor = self.batch_padding(input_tensor)

            ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training)

            self.reset_variables()
        else:
            #ret_val = self.type_call['ANN'](input_tensor, f_training)
            ret_val = self.type_call[self.conf.nn_mode](input_tensor, f_training)

            #if self.conf.nn_mode=='SNN':
            #    ret_val = self.type_call['SNN'](input_tensor, f_training)
            self.f_model_load_done = True
        return ret_val


    def call_ann(self, inputs, f_training):
        #self.inputs = inputs
        self.list_a[0] = inputs

        #s_h = self.fc_h(inputs)
        self.list_s[0] = self.list_l[0](self.list_a[0])
        #a_h = tf.nn.relu(s_h)
        #self.list_a[0] = self.list_n[0](self.s_h)
        self.list_a[1] = self.list_n[0](self.list_s[0])

        #s_o = self.fc_o(a_h)
        self.list_s[1] = self.list_l[1](self.list_a[1])
        #self.s_o = self.list_l[1](self.a_h)
        #self.list_a[1] = self.s_o
        self.list_a[2] = self.list_s[1]
        #a_o = tf.nn.relu(s_o)
        #a_o = self.list_n[1](s_o)

        ret_val = self.list_a[2]

        return ret_val

    def recording_ret_val(self, output_neuron):
        output_neuron = output_neuron
        # spike count
        #print(self.o_shape)
        #print(tf.shape(self.spike_count))
        #print(self.count_accuracy_time_point)
        #print(tf.shape(output_neuron.get_spike_count()))
        self.spike_count[self.count_accuracy_time_point,:,:]=(output_neuron.get_spike_count().numpy())
        # vmem
        #spike_count[count_accuracy_time_point,:,:]=(output_neuron.vmem.numpy())

        # get total spike count
        #tc_int, tc = self.get_total_spike_count()
        #
        #self.total_spike_count_int[count_accuracy_time_point]+=tc_int
        #self.total_spike_count[count_accuracy_time_point]+=tc

        self.count_accuracy_time_point+=1


        #num_spike_count = tf.cast(tf.reduce_sum(spike_count,axis=[2]),tf.int32)



    def call_snn(self, inputs, f_training):
        if self.f_model_load_done:
            self.reset_neuron()

        for t in range(self.conf.time_step):
            self.list_a[0] = self.list_n[0](inputs,t)
            self.list_s[0] = self.list_l[0](self.list_a[0])

            #if self.f_model_load_done:
            #    psp = tf.expand_dims(self.list_a[0],2)
            #    psp = tf.tile(psp, [1,1,tf.shape(self.list_l[0].kernel)[1]])
            #    psp = tf.multiply(self.list_l[0].kernel,psp)
            #    self.list_tpsp[0] = tf.add(self.list_tpsp[0],psp)

            self.list_a[1] = self.list_n[1](self.list_s[0],t)
            self.list_s[1] = self.list_l[1](self.list_a[1])

            #if self.f_model_load_done:
            #    psp = tf.expand_dims(self.list_a[1],2)
            #    psp = tf.tile(psp, [1,1,tf.shape(self.list_l[1].kernel)[1]])
            #    psp = tf.multiply(self.list_l[1].kernel,psp)
            #    self.list_tpsp[1] = tf.add(self.list_tpsp[1],psp)

            self.list_a[2] = self.list_n[2](self.list_s[1],t)

            if t==self.accuracy_time_point[self.count_accuracy_time_point]-1:
               self.recording_ret_val(self.list_n[2])


        #ret_val = self.list_n[1].get_spike_count()/self.conf.time_step    # rate-coding
        ret_val = self.spike_count/self.conf.time_step

        return ret_val


    # for neuron
    def reset_neuron(self):

        self.count_accuracy_time_point = 0
        self.spike_count = np.zeros((self.num_accuracy_time_point,)+self.o_shape)

        for idx_n, n in enumerate(self.list_n):
            n.reset()

