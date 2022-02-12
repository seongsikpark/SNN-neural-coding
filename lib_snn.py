import tensorflow as tf

import numpy as np

class Neuron(tf.keras.layers.Layer):
    def __init__(self,dim,n_type,fan_in,conf,neural_coding,depth=0,n_name='',**kwargs):
        super(Neuron, self).__init__()

        self.dim = dim
        self.dim_one_batch = [1,]+dim[1:]

        self.n_type = n_type
        self.fan_in = fan_in

        self.conf = conf

        self.neural_coding=neural_coding

        self.n_name = n_name

        # stat for weighted spike
        self.en_stat_ws = False

        self.zeros = tf.zeros(self.dim,dtype=tf.float32)
        self.fires = tf.constant(self.conf.n_in_init_vth,shape=self.dim,dtype=tf.float32)

        self.depth = depth

        self.init_first_spike_time = self.conf.time_fire_duration

        if self.conf.neural_coding=='TEMPORAL':
            self.time_const_init_fire = self.conf.tc
            self.time_const_init_integ = self.conf.tc

            self.time_delay_init_fire = 0.0
            self.time_delay_init_integ = 0.0

            self.time_start_integ_init = (self.depth-1)*self.conf.time_fire_start
            self.time_start_fire_init = (self.depth)*self.conf.time_fire_start

            self.time_end_integ_init = self.time_start_integ_init + self.conf.time_fire_duration
            self.time_end_fire_init = self.time_start_fire_init + self.conf.time_fire_duration

    def build(self, _):

        if self.n_type == 'IN':
            init_vth = self.conf.n_in_init_vth
        else:
            init_vth = self.conf.n_init_vth

        self.vth_init = self.add_variable("vth_init",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(init_vth),trainable=False)
        self.vth = self.add_variable("vth",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(init_vth),trainable=False)
        self.vmem = self.add_variable("vmem",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.conf.n_init_vinit),trainable=False)
        self.out = self.add_variable("out",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.spike_counter_int = self.add_variable("spike_counter_int",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
        self.spike_counter = self.add_variable("spike_counter",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.f_fire = self.add_variable("f_fire",shape=self.dim,dtype=tf.bool,trainable=False)

        if self.conf.f_tot_psp:
            self.tot_psp = self.add_variable("tot_psp",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        if self.conf.f_refractory:
            self.refractory = self.add_variable("refractory",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
            self.t_set_refractory= self.add_variable("t_set_refractory",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(-1.0),trainable=False)

        # stat for weighted spike
        if self.en_stat_ws:
            self.stat_ws = self.add_variable("stat_ws",shape=self.conf.p_ws,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.first_spike_time=self.add_variable("first_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.init_first_spike_time),trainable=False)
        if self.conf.neural_coding=='TEMPORAL':
            # TODO: old - scalar version
            self.time_const_integ=self.add_variable("time_const_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_integ),trainable=False)
            self.time_const_fire=self.add_variable("time_const_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_fire),trainable=False)
            self.time_delay_integ=self.add_variable("time_delay_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_integ),trainable=False)
            self.time_delay_fire=self.add_variable("time_delay_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_fire),trainable=False)

            self.time_start_integ=self.add_variable("time_start_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_integ_init),trainable=False)
            self.time_end_integ=self.add_variable("time_end_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_integ_init),trainable=False)
            self.time_start_fire=self.add_variable("time_start_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_fire_init),trainable=False)
            self.time_end_fire=self.add_variable("time_end_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_fire_init),trainable=False)

            print_loss=True

            if self.conf.f_train_time_const and print_loss:
                self.loss_prec=self.add_variable("loss_prec",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
                self.loss_min=self.add_variable("loss_min",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
                self.loss_max=self.add_variable("loss_max",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

    def call(self,inputs,t):

        # run_fwd
        run_type = {
            'IN': self.run_type_in,
            'IF': self.run_type_if,
            'LIF': self.run_type_lif,
            'OUT': self.run_type_out
        }[self.n_type](inputs,t)

        out_ret = self.out

        return out_ret

    # reset - time step
    def reset_each_time(self):
        self.reset_out()

    # reset - sample
    def reset(self):
        #print('reset neuron')
        self.reset_vmem()
        self.reset_spike_count()
        self.reset_vth()

        if self.conf.f_tot_psp:
            self.reset_tot_psp()
        if self.conf.f_refractory:
            self.refractory = tf.zeros(self.refractory.shape)
            self.t_set_refractory = tf.constant(-1.0,dtype=tf.float32,shape=self.refractory.shape)

        if self.conf.f_record_first_spike_time:
            self.reset_first_spike_time()

    def reset_spike_count(self):
        self.spike_counter = tf.zeros(self.dim,dtype=tf.float32)
        self.spike_counter_int = tf.zeros(self.dim,dtype=tf.float32)

    #
    def reset_vmem(self):
        self.vmem = tf.constant(self.conf.n_init_vinit,tf.float32,self.vmem.shape)

    #
    def reset_tot_psp(self):
        self.tot_psp = tf.zeros(tf.shape(self.tot_psp))

    #
    def reset_out(self):
        self.out = tf.zeros(self.out.shape)

    def reset_vth(self):
        self.vth = self.vth_init

    #
    def reset_first_spike_time(self):
        self.first_spike_time=tf.constant(self.init_first_spike_time,shape=self.first_spike_time.shape,dtype=tf.float32)

    ##
    def set_vth_temporal_kernel(self,t):
        # exponential decay
        time = tf.subtract(t,self.time_delay_fire)
        self.vth = tf.multiply(self.conf.n_init_vth,tf.exp(tf.divide(-time,self.time_const_fire)))

    ##
    def input_spike_real(self,inputs,t):
        # TODO: check it
        if self.conf.neural_coding=="WEIGHTED_SPIKE":
            self.out=tf.truediv(inputs,self.conf.p_ws)
        elif self.conf.neural_coding=="TEMPORAL":
            if t==0:
                self.out=inputs
            else:
                self.out = tf.zeros(self.out.shape)
        else:
            self.out=inputs

    def input_spike_poission(self,inputs,t):
        # Poission input
        vrand = tf.random_uniform(self.vmem.shape,minval=0.0,maxval=1.0,dtype=tf.float32)
        self.f_fire = inputs>=vrand
        self.out = tf.where(self.f_fire,self.fires,self.zeros)

    def input_spike_weighted_spike(self,inputs,t):
        # weighted synpase input
        t_mod = (int)(t%8)
        if t_mod == 0:
            self.vmem = inputs
            self.vth = tf.constant(0.5,tf.float32,self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth,0.5)

        f_fire = self.vmem >= self.vth
        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem,self.out)

    def input_spike_burst(self,inputs,t):
        if t == 0:
            self.vmem = inputs

        f_fire = self.vmem >= self.vth
        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem,self.out)
        self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)

        # repeat input
        if tf.equal(tf.reduce_max(self.out),0.0):
            self.vmem = inputs

    def input_spike_temporal(self,inputs,t):
        if t == 0:
            self.vmem = inputs

        self.set_vth_temporal_kernel(t)

        if self.conf.f_refractory:
            self.f_fire = (self.vmem >= self.vth) & \
                          tf.equal(self.refractory,tf.constant(0.0,tf.float32,self.refractory.shape))
        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10**(-5))

        # reset by subtraction
        #self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        #self.vmem = tf.subtract(self.vmem,self.out)

        # reset by zero
        self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)

        if self.conf.f_refractory:
            self.cal_refractory_temporal(t)

    def spike_dummy_input(self,inputs,t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False,tf.bool,self.f_fire.shape)

    def spike_dummy_fire(self,t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False,tf.bool,self.f_fire.shape)

    # (min,max) = (0.0,1.0)
    def input_spike_gen(self,inputs,t):

        f_run_temporal = (t < self.conf.time_fire_duration)

        input_spike_mode ={
            'REAL': self.input_spike_real,
            'POISSON': self.input_spike_poission,
            'WEIGHTED_SPIKE': self.input_spike_weighted_spike,
            'BURST': self.input_spike_burst,
            'TEMPORAL': self.input_spike_temporal if f_run_temporal else self.spike_dummy_input
        }
        input_spike_mode[self.conf.input_spike_mode](inputs,t)

    #
    def leak(self):
        self.vmem = tf.multiply(self.vmem,0.7)

    #
    def integration(self,inputs,t):

        if self.conf.neural_coding=="TEMPORAL":
            t_int_s = self.time_start_integ_init
            t_int_e = self.time_end_integ_init

            f_run_int_temporal = (t >= t_int_s and t < t_int_e) or (t==0)   # t==0 : for bias integration
        else:
            f_run_int_temporal = False

        # intergation
        {
            'TEMPORAL': self.integration_temporal if f_run_int_temporal else lambda inputs, t : None
        }.get(self.neural_coding, self.integration_default) (inputs,t)


        ########################################
        # common
        ########################################
        #
        if self.conf.f_positive_vmem:
            self.vmem = tf.maximum(self.vmem,tf.constant(0.0,tf.float32,self.vmem.shape))
        #
        if self.conf.f_tot_psp:
            self.tot_psp = tf.add(self.tot_psp, inputs)

    #
    def integration_default(self,inputs,t):
        self.vmem = tf.add(self.vmem,inputs)

    #
    def integration_temporal(self,inputs,t):
        if(t==0) :
            time = 0
        else :
            time = self.relative_time_integ(t)
            time = time - self.time_delay_integ
        receptive_kernel = tf.exp(tf.divide(-time,self.time_const_integ))

        if self.depth==1:
            if self.conf.input_spike_mode=='REAL':
                psp = inputs
            else:
                psp = tf.multiply(inputs,receptive_kernel)
        else:
            psp = tf.multiply(inputs,receptive_kernel)

        #
        self.vmem = tf.add(self.vmem,psp)

    ############################################################
    ## fire function
    ############################################################
    def fire(self,t):

        #
        # for TTFS coding
        #
        if self.conf.neural_coding=="TEMPORAL":
            t_fire_s = self.time_start_fire_init
            t_fire_e = self.time_end_fire_init
            f_run_fire = t >= t_fire_s and t < t_fire_e
        else:
            f_run_fire = False

        {
            'RATE': self.fire_rate,
            'WEIGHTED_SPIKE': self.fire_weighted_spike,
            'BURST': self.fire_burst,
            'TEMPORAL': self.fire_temporal if f_run_fire else self.spike_dummy_fire,
        }.get(self.neural_coding, self.fire_rate) (t)



    #
    def fire_rate(self,t):
        self.f_fire = self.vmem >= self.vth
        self.out = tf.where(self.f_fire,self.fires,self.zeros)

        # reset by subtraction
        self.vmem = tf.subtract(self.vmem,self.out)

        # reset to zero
        #self.vmem = tf.where(f_fire,tf.constant(self.conf.n_init_vreset,tf.float32,self.vmem.shape,self.vmem)



    def fire_weighted_spike(self,t):
        # weighted synpase input
        t_mod = (int)(t%self.conf.p_ws)
        if t_mod == 0:
            self.vth = tf.constant(self.conf.n_init_vth,tf.float32,self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth,0.5)

        if self.conf.f_refractory:
            assert False, 'modify refractory'
        else:
            self.f_fire = self.vmem >= self.vth

        if self.conf.f_refractory:
            print('fire_weighted_spike, refractory: not implemented yet')

        self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem,self.out)

        # stat for weighted spike
        if self.en_stat_ws:
            count = tf.cast(tf.math.count_nonzero(self.out),tf.float32)
            self.stat_ws.scatter_add(tf.IndexedSlices(count,t_mod))

    #
    def fire_burst(self,t):
        if self.conf.f_refractory:
            assert False, 'modify refractory'
        else:
            self.f_fire = self.vmem >= self.vth

        # reset by subtraction
        self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem,self.out)

        if self.conf.f_refractory:
            self.cal_refractory(self.f_fire)

        # exp increasing order
        self.vth = tf.where(self.f_fire,self.vth*2.0,self.vth_init)

    #
    def fire_temporal(self,t):
        time = self.relative_time_fire(t)

        #
        # encoding
        # dynamic threshold (vth)
        #
        self.set_vth_temporal_kernel(time)

        if self.conf.f_refractory:
            f_fire = tf.greater_equal(self.vmem,self.vth)
            f_refractory = tf.greater(tf.constant(t,tf.float32),self.refractory)
            self.f_fire = tf.logical_and(f_fire,f_refractory)
        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10**(-5))

        #
        # reset
        #
        # reset by zero
        self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)
        # reset by subtraction
        #self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        #self.vmem = tf.subtract(self.vmem,self.out)

        if self.conf.f_refractory:
            self.cal_refractory_temporal(t)

    def fire_type_out(self, t):
        f_fire = self.vmem >= self.vth
        self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)
        self.out = tf.where(f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))


    ############################################################
    ##
    ############################################################

    def cal_refractory(self,f_fire):
        f_refractory_update = np.logical_and(np.not_equal(self.vth-self.vth_init,0.0),np.logical_not(f_fire))
        refractory_update = 2.0*np.log2(self.vth/self.vth_init)
        self.refractory = tf.maximum(self.refractory-1,tf.constant(0.0,tf.float32,self.refractory.shape))
        self.refractory = tf.where(f_refractory_update,refractory_update,self.refractory)

    def cal_refractory_temporal_original(self,f_fire):
        self.refractory = tf.where(f_fire,tf.constant(10000.0,tf.float32,self.refractory.shape),self.refractory)

    def cal_refractory_temporal(self, t):
        self.refractory = tf.where(self.f_fire,tf.constant(10000.0,tf.float32,self.refractory.shape),self.refractory)

    #
    def count_spike(self, t):
        {
            'TEMPORAL': self.count_spike_temporal
        }.get(self.neural_coding, self.count_spike_default) (t)


    def count_spike_default(self, t):
        self.spike_counter_int = tf.where(self.f_fire,self.spike_counter_int+1.0,self.spike_counter_int)
        self.spike_counter = tf.add(self.spike_counter, self.out)

    def count_spike_temporal(self, t):
        self.spike_counter_int = tf.add(self.spike_counter_int, self.out)
        self.spike_counter = tf.where(self.f_fire, tf.add(self.spike_counter,self.vth),self.spike_counter_int)

        if self.conf.f_record_first_spike_time:
            spike_time = t

            self.first_spike_time = tf.where(
                                        self.f_fire,\
                                        tf.constant(spike_time,dtype=tf.float32,shape=self.first_spike_time.shape),\
                                        self.first_spike_time)

    ############################################################
    ## run fwd pass
    ############################################################

    def run_type_in(self,inputs,t):
        #print('run_type_in')
        self.input_spike_gen(inputs,t)
        self.count_spike(t)

    #
    def run_type_if(self,inputs,t):
        #print('run_type_if')
        self.integration(inputs,t)
        self.fire(t)
        self.count_spike(t)

    #
    def run_type_lif(self,inputs,t):
        #print('run_type_lif')
        self.leak()
        self.integration(inputs,t)
        self.fire(t)
        self.count_spike(t)

    def run_type_out(self,inputs,t):
        #print("output layer")

        if self.conf.snn_output_type in ('SPIKE', 'FIRST_SPIKE_TIME'):
            # in current implementation, output layer acts as IF neuron.
            # If the other types of neuron is needed for the output layer,
            # the declarations of neuron layers in other files should be modified.
            self.run_type_if(inputs,t)
        else:
            self.integration(inputs,t)


    ############################################################
    ##
    ############################################################

    def set_vth(self,vth):
        self.vth.assign(vth)

    def get_spike_count(self):
        spike_count = tf.reshape(self.spike_counter,self.dim)
        return spike_count

    def get_spike_count_int(self):
        return self.spike_counter_int

    def get_spike_rate(self):
        return self.get_spike_count_int()/self.conf.time_step

    def get_tot_psp(self):
        return self.tot_psp

    def set_time_const_init_integ(self, time_const_init_integ):
        self.time_const_init_integ = time_const_init_integ

    def set_time_const_init_fire(self, time_const_init_fire):
        self.time_const_init_fire = time_const_init_fire

    def set_time_delay_init_integ(self, time_delay_init_integ):
        self.time_delay_init_integ = time_delay_init_integ

    def set_time_delay_init_fire(self, time_delay_init_fire):
        self.time_delay_init_fire = time_delay_init_fire

    def set_time_const_integ(self, time_const_integ):
        self.time_const_integ = time_const_integ

    def set_time_const_fire(self, time_const_fire):
        self.time_const_fire = time_const_fire

    def set_time_delay_integ(self, time_delay_integ):
        self.time_delay_integ = time_delay_integ

    def set_time_delay_fire(self, time_delay_fire):
        self.time_delay_fire = time_delay_fire

    def set_time_integ(self, time_start_integ):
        self.time_start_integ = time_start_integ
        self.time_end_integ = self.time_start_integ + self.conf.time_fire_duration

    def set_time_fire(self, time_start_fire):
        self.time_start_fire = time_start_fire
        self.time_end_fire = self.time_start_fire + self.conf.time_fire_duration


    ############################################################
    ## training time constant (tau) for TTFS coding
    ## gradient-based optimization (DAC-20)
    ############################################################
    def train_time_const_fire(self,dnn_act):
        #print("snn_lib: train_time_const")
        #print(dnn_act)
        #self.time_const_integ = tf.zeros([])
        #self.time_const_fire = tf.multiply(self.time_const_fire,0.1)

        # delta - -1/(2tau^2)(x-x_hat)(x_hat)

        #spike_time = self.first_spike_time-self.depth*self.conf.time_fire_start*self.time_const_fire-self.time_delay_fire


        spike_time = self.first_spike_time
        spike_time_sub_delay = spike_time-self.time_delay_fire


        x = dnn_act

        #x_hat = tf.where(
        #            tf.equal(self.first_spike_time,tf.constant(self.init_first_spike_time,shape=self.first_spike_time.shape,dtype=tf.float32)), \
        #            tf.zeros(self.first_spike_time.shape), \
        #            tf.exp(-(spike_time_sub_delay/self.time_const_fire)))


        x_hat = tf.where(
                    self.flag_fire(),\
                    tf.zeros(self.first_spike_time.shape), \
                    tf.exp(-(spike_time_sub_delay/self.time_const_fire)))


        #x_hat = tf.exp(-self.first_spike_time/self.time_const_fire)

        #loss = tf.reduce_sum(tf.square(x-x_hat))


        loss_prec = tf.reduce_mean(tf.square(x-x_hat))
        loss_prec = loss_prec/2.0

        # l2
        delta1 = tf.subtract(x,x_hat)
        delta1 = tf.multiply(delta1,x_hat)
        #delta1 = tf.multiply(delta1, tf.subtract(self.first_spike_time,self.time_delay_fire))
        delta1 = tf.multiply(delta1, spike_time_sub_delay)

        if tf.equal(tf.size(tf.boolean_mask(delta1,delta1>0)),0):
            delta1 = tf.zeros([])
        else:
            delta1 = tf.reduce_mean(tf.boolean_mask(delta1,delta1>0))


        dim = tf.size(tf.shape(x))
        if tf.equal(dim, tf.constant(4)):
            reduce_axis = [1,2,3]
        elif tf.equal(dim, tf.constant(3)):
            reduce_axis = [1,2]
        elif tf.equal(dim, tf.constant(2)):
            reduce_axis = [1]


        if self.conf.f_train_time_const_outlier:
            #x_min = tf.tfp.stats.percentile(tf.boolean_mask(x,x>0),0.01)
            #x_min = tf.constant(np.percentile(tf.boolean_mask(x,x>0).numpy(),1),dtype=tf.float32,shape=[])

            x_pos = tf.where(x>tf.zeros(x.shape),x,tf.zeros(x.shape))
            x_min = tf.constant(np.percentile(x_pos.numpy(),2,axis=reduce_axis),dtype=tf.float32,shape=x_pos.shape[0])

            #print("min: {:e}, min_0.01: {:e}".format(tf.reduce_min(tf.boolean_mask(x,x>0)),x_min))
        else:
            #~x_min = tf.reduce_min(tf.boolean_mask(x,x>0))
            x_pos = tf.where(x>tf.zeros(x.shape),x,tf.zeros(x.shape))
            x_min = tf.reduce_min(x_pos,axis=reduce_axis)

        fire_duration = self.conf.time_fire_duration

        #x_hat_min = tf.exp(-(self.conf.time_fire_duration/self.time_const_fire))
        x_hat_min = tf.exp(-(fire_duration-self.time_delay_fire)/self.time_const_fire)

        loss_min = tf.reduce_mean(tf.square(x_min-x_hat_min))
        loss_min = loss_min/2.0

        x_min = tf.reduce_mean(x_min)




        delta2 = tf.subtract(x_min,x_hat_min)
        delta2 = tf.multiply(delta2,x_hat_min)

        delta2 = tf.multiply(delta2, tf.subtract(self.conf.time_window,self.time_delay_fire))


        #delta2 = tf.reduce_mean(delta2)

        #
        #idx=0,0,0,0
        #print("x: {:e}, x_hat: {:e}".format(x[idx],x_hat[idx]))
        #print("x_min: {:e}, x_hat_min: {:e}".format(x_min,x_hat_min))

        # l1


        #
        delta1 = tf.divide(delta1,tf.square(self.time_const_fire))
        delta2 = tf.divide(delta2,tf.square(self.time_const_fire))

        # learning rate
        #rho1 = 10.0
        #rho2 = 100.0

        rho1 = 10.0
        rho2 = 100.0

        #
        delta = tf.add(tf.multiply(delta1,rho1),tf.multiply(delta2,rho2))

        #print("name: {:s}, del: {:e}, del1: {:e}, del2: {:e}".format(self.n_name,delta,delta1,delta2))


        #self.time_const_fire = tf.subtract(self.time_const_fire, delta)
        self.time_const_fire = tf.add(self.time_const_fire, delta)


        #
        #idx=0,10,10,0
        #print('x: {:e}, vmem: {:e}, x_hat: {:e}, delta: {:e}'.format(x[idx],self.vmem[idx],x_hat[idx],delta))

        print("name: {:s}, loss_prec: {:g}, loss_min: {:g}, tc: {:f}".format(self.n_name,loss_prec,loss_min,self.time_const_fire))

        self.loss_prec = loss_prec
        self.loss_min = loss_min

        #
        #print("name: {:s}, tc: {:f}".format(self.n_name,self.time_const_fire))


        #print("\n")

    ############################################################
    ## training time delay (td) for TTFS coding
    ## gradient-based optimization (DAC-20)
    ############################################################
    def train_time_delay_fire(self, dnn_act):

#        if self.conf.f_train_time_const_outlier:
#            t_ref = self.depth*self.conf.time_fire_start
#            t_min = np.percentile(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0).numpy(),0.01)
#            t_min = t_min-t_ref
#        else:
#            t_ref = self.depth*self.conf.time_fire_start
#            t_min = tf.reduce_min(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0))
#            t_min = t_min-t_ref


        #t_ref = self.depth*self.conf.time_fire_start*self.time_const_fire

        t_ref = self.depth*self.conf.time_fire_start


        dim = tf.size(tf.shape(self.first_spike_time))
        if tf.equal(dim, tf.constant(4)):
            reduce_axis = [1,2,3]
        elif tf.equal(dim, tf.constant(3)):
            reduce_axis = [1,2]
        elif tf.equal(dim, tf.constant(2)):
            reduce_axis = [1]

        #print(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0,keepdims=True).shape)

        #t_min = tf.reduce_min(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0))
        t_min = tf.where(tf.equal(self.first_spike_time,self.init_first_spike_time),tf.constant(99999.9,shape=self.first_spike_time.shape),self.first_spike_time)
        t_min = tf.reduce_min(t_min,axis=reduce_axis)
        #t_min = t_min-t_ref
        t_min = self.relative_time_fire(t_min)

        #print(self.depth)
        #print(t_min)

        x_max = tf.exp(-(t_min-self.time_delay_fire)/self.time_const_fire)

        x_max_hat = tf.exp(self.time_delay_fire/self.time_const_fire)

        loss_max = tf.reduce_mean(tf.square(x_max-x_max_hat))
        loss_max = loss_max/2.0

        delta = tf.subtract(x_max,x_max_hat)
        delta = tf.multiply(delta,x_max_hat)
        delta = tf.divide(delta,self.time_const_fire)
        delta = tf.reduce_mean(delta)

        # learning rate
        rho = 1.0
        #rho = 0.01

        delta = tf.multiply(delta,rho)

        #self.time_delay_fire = tf.subtract(self.time_delay_fire,delta)
        self.time_delay_fire = tf.add(self.time_delay_fire,delta)

        #print("name: {:s}, del: {:e}, td: {:e}".format(self.n_name,delta,self.time_delay_fire))
        print("name: {:s}, loss_max: {:e}, td: {:f}".format(self.n_name,loss_max,self.time_delay_fire))

        self.loss_max = loss_max


    ############################################################
    ## This function is needed for fire phase in the temporal coding (TTFS)
    ## time converter: absolute time (global time) -> relative time in each time window (local time)
    ## t: absolute time (global time)
    ## time: relative time (local time)
    ############################################################
    def relative_time_fire(self, t):
        time = t-self.depth*self.conf.time_fire_start
        return time

    ############################################################
    ## This function is needed for integration phase in the temporal coding (TTFS)
    ## time converter: absolute time (global time) -> relative time in each time window (local time)
    ## t: absolute time (global time)
    ## time: relative time (local time)
    ############################################################
    def relative_time_integ(self, t):
        time = t-(self.depth-1)*self.conf.time_fire_start
        return time


    #
    def flag_fire(self):
        ret = tf.not_equal(self.spike_counter,tf.constant(0.0,tf.float32,self.spike_counter.shape))
        return ret





############################################################
## spike max pool (spike count based gating function)
############################################################
def spike_max_pool(feature_map, spike_count, output_shape):
    tmp = tf.reshape(spike_count,(1,-1,)+tuple(tf.shape(spike_count)[2:]))
    _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
    conv_f = tf.reshape(feature_map,[-1])
    arg = tf.reshape(arg,[-1])

    p_conv = tf.gather(conv_f, arg)
    p_conv = tf.reshape(p_conv,output_shape)

    return p_conv

