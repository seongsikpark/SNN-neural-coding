from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe

from tensorflow.python.framework import ops

import functools
import itertools

#import backprop as bp_sspark

#

#

#
cce = tf.keras.losses.CategoricalCrossentropy()

# TODO: remove
def loss(predictions, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=labels))


def loss_cross_entoropy(predictions, labels):

    logits = tf.nn.softmax(predictions)
    loss_value = cce(labels,logits)

    return loss_value



def loss_mse(predictions, labels):
    loss_value = tf.losses.mean_squared_error(labels, predictions)
    delta = predictions - labels

    return loss_value, delta



#
def compute_accuracy(predictions, labels):
    return tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.argmax(predictions, axis=1, output_type=tf.int64),
                        tf.argmax(labels, axis=1, output_type=tf.int64)
                    ), dtype=tf.float32
                )
            ) / float(predictions.shape[0].value)



#@profile
def train_one_epoch_original_ref(model, optimizer, dataset):
    #tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')

    def model_loss(labels, images):
        prediction = model(images, f_training=True)
        loss_value = loss(prediction, labels)
        avg_loss(loss_value)
        accuracy(tf.argmax(prediction,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
        #tf.contrib.summary.scalar('loss', loss_value)
        #tf.contrib.summary.scalar('accuracy', compute_accuracy(prediction, labels))
        return loss_value

    #builder = tf.profiler.ProfileOptionBuilder
    #opts = builder(builder.time_and_memory()).order_by('micros').build()

    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:
        #print(batch)
        #print(images)
        #print(labels)
        #with tf.contrib.summary.record_summaries_every_n_global_steps(10):

        batch_model_loss = functools.partial(model_loss, labels, images)
        optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())

        #batch_model_loss = functools.partial(model_loss, labels, images)
        #optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())

    #print('Train set: Accuracy: %4f%%\n'%(100*accuracy.result()))
    return avg_loss.result(), 100*accuracy.result()



#
def train_one_epoch(mode_tmp):

    # Todo: choose training function depending on dataset and model
    train_func_sel = {
        'ANN': train_one_epoch_mnist_cnn
        #'ANN': train_one_epoch_original_ref
    }
    train_func=train_func_sel[mode_tmp]

    return train_func



###############################################################################
##
###############################################################################
def train_one_epoch_mnist_cnn(model, optimizer, dataset):
    #tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')


    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:

            #tape.watch(images)
            #tape.watch(model.fc1.kernel)

            predictions = model(images, f_training=True)

            #loss_value = loss(predictions, labels)
            loss_value = loss_cross_entoropy(predictions, labels)
            avg_loss(loss_value)

            accuracy(tf.argmax(predictions,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))

        #print(grad)



        if False:
            for layer_name in model.layer_name:

                layer=model.list_layer[layer_name]

                grad = tape.gradient(loss_value, layer.kernel)
                grad_b = tape.gradient(loss_value, layer.bias)

                grads_and_vars.append((grad,layer.kernel))
                grads_and_vars.append((grad_b,layer.bias))

        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss_value, trainable_vars)
        #print(trainable_vars)

        grads_and_vars = zip(grads,trainable_vars)

        optimizer.apply_gradients(grads_and_vars)




    return avg_loss.result(), 100*accuracy.result()


###############################################################################
##
###############################################################################


def train_one_epoch_mnist_mlp(model, optimizer, dataset):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')

    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, f_training=True)

            loss_value = loss(predictions, labels)
            avg_loss(loss_value)

            accuracy(tf.argmax(predictions,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        # backprop
        delta = []
        delta_tmp = loss_value - labels
        delta.append(delta_tmp)

        #delta_tmp = bp_Dense(model.list_l[3],delta_tmp)
        #delta_tmp = bp_relu(model.list_a[3], delta_tmp)
        #delta.append(delta_tmp)


        # gradient
        layer = model.list_l[3]
        d_local = model.list_a[3]
        delt = delta[0]

        #print(tf.shape(delt))
        #print(tf.shape(d_local))
        grad_t, grad_b_t = grad_Dense(layer,delt,d_local)

        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)




        del_in = delta_tmp
        d_s_w = d_local

        del_in_e = tf.tile(tf.expand_dims(del_in,1), [1,tf.shape(layer.kernel)[0],1])
        d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])

        grad_t = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)
        grad_b_t = tf.reduce_sum(del_in,0)


        #grad_b = tape.gradient(loss_value,model.list_a[4])

        #print(grad_b_t)
        #3print(grad_b)

        #diff = grad_t - grad
        #diff = grad_b_t - grad_t
        #print('max: %.3f, sum: %.3f'%(tf.reduce_max(diff),tf.reduce_sum(diff)))



        grads_and_vars.append((grad,layer.kernel))
        grads_and_vars.append((grad_b,layer.bias))

        layer = model.list_l[2]
        #d_local = model.p_flat
        #delt = delta[1]
        #grad, grad_b = grad_Dense(layer,delt,d_local)
        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)

        grads_and_vars.append((grad,layer.kernel))
        grads_and_vars.append((grad_b,layer.bias))


        layer = model.list_l[1]
        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)

        grads_and_vars.append((grad, layer.kernel))
        grads_and_vars.append((grad_b, layer.bias))


        layer = model.list_l[0]
        grad = tape.gradient(loss_value, layer.kernel)
        grad_b = tape.gradient(loss_value, layer.bias)

        grads_and_vars.append((grad, layer.kernel))
        grads_and_vars.append((grad_b, layer.bias))


        optimizer.apply_gradients(grads_and_vars)

        #grads = tape.gradient(loss_value, model.variables)
        #optimizer.apply_gradients(zip(grads, model.variables))

    return avg_loss.result(), 100*accuracy.result()

def train_snn_one_epoch_mnist_cnn(model, optimizer, dataset, conf):
    print('not defined yet')

def train_ann_one_epoch_mnist(model, optimizer, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')


    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, f_training=True)


            #loss_value, delta = loss_cross_entropy(predictions, labels)
            loss_value, delta = loss_mse(predictions, labels)

            avg_loss(loss_value)

            accuracy(tf.argmax(predictions,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        #del_last = loss_value - labels_expandded

        #softmax = tf.nn.softmax(predictions)

        #del_last = tf.cast(softmax,tf.float32) - tf.cast(labels,tf.float32)
        #del_last = tf.cast(predictions,tf.float32) - tf.cast(labels,tf.float32)

        del_last = delta

        idx_layer_last = 1

        for idx_layer in range(idx_layer_last,-1,-1):

            layer = model.list_l[idx_layer]
            act = model.list_a[idx_layer]
            neuron = model.list_n[idx_layer]

            # backprop
            if idx_layer == idx_layer_last:
                del_in = del_last
            else:
                del_in = del_out

            del_out = bp_Dense(layer, del_in)
            del_out = bp_relu(act, del_out)

            #print(tape.gradient(model.list_s[-1], model.list_a[-1]))
            #print(tape.gradient(loss_value,model.list_a[-1]))


            #test = tfe.gradients_function(model.list_l[1].call)
            #test = tfe.implicit_gradients(model.list_l[1].call)

            #print(test(del_in))


            # weight update
            d_s_w = act

            del_in_e = tf.tile(tf.expand_dims(del_in,1), [1,tf.shape(layer.kernel)[0],1])
            d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])

            grad_b = tf.reduce_sum(del_in,0)
            grad = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)

            grads_and_vars.append((grad, layer.kernel))
            grads_and_vars.append((grad_b, layer.bias))

        optimizer.apply_gradients(grads_and_vars)

    return avg_loss.result(), 100*accuracy.result()



def train_snn_one_epoch_mnist_psp_ver(model, optimizer, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')

    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:
        grads_and_vars = []

        #with tf.GradientTape(persistent=True) as tape:
        predictions_times = model(images, f_training=True)

        if predictions_times.shape[1] != labels.numpy().shape[0]:
            predictions_times_trimmed = predictions_times[:,0:labels.numpy().shape[0],:]
            d = predictions_times.shape[1] - labels.numpy().shape[0]

            labels_expandded = tf.concat([labels, tf.zeros((d,)+(labels.numpy().shape[1],))],axis=0)
            f_exp = True
        else:
            predictions_times_trimmed = predictions_times
            labels_expandded = labels
            f_exp = False


        predictions_trimmed = predictions_times_trimmed[-1]

        loss_value = loss(predictions_trimmed, labels)

        avg_loss(loss_value)

        #accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
        accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        #predictions = predictions_times[-1]

        #del_last = loss_value - labels_expandded

        softmax = tf.nn.softmax(predictions_trimmed)

        del_last = tf.cast(softmax,tf.float32) - tf.cast(labels,tf.float32)
        #del_last = tf.cast(predictions_trimmed,tf.float32) - tf.cast(labels,tf.float32)

        if f_exp :
            del_last = tf.concat([del_last, tf.zeros((d,)+(del_last.numpy().shape[1],))],axis=0)

        idx_layer_last = 1

        for idx_layer in range(idx_layer_last,-1,-1):

            layer = model.list_l[idx_layer]
            act = model.list_a[idx_layer]
            neuron_pri = model.list_n[idx_layer]
            neuron_post = model.list_n[idx_layer+1]
            tpsp = model.list_tpsp[idx_layer]

            # backprop
            if idx_layer == idx_layer_last:
                del_in = del_last
            else:
                del_in = del_out

            del_out = bp_Dense(layer, del_in)
            #del_out = tf.multiply(del_out,neuron.get_spike_rate())
            #del_out = bp_relu(act, del_out)
            tpsp_m_post = tf.reduce_sum(tpsp,2)
            tpsp_m_post = tf.divide(tf.cast(tpsp_m_post,tf.float32),tf.cast(neuron_pri.spike_counter_int*conf.n_init_vth,tf.float32))
            del_out = tf.multiply(del_out,tpsp_m_post)

            #print(tape.gradient(model.list_s[-1], model.list_a[-1]))
            #print(tape.gradient(loss_value,model.list_a[-1]))
            #print(tape.gradient(loss_value,model.list_l[-1].kernel))


            # weight update
            #d_s_w = act
            #d_s_w = neuron.get_spike_rate()
            #d_s_w = neuron.get_tot_psp()/neuron.get_spike_count_int()
            #d_s_w = neuron.get_spike_rate()

            tpsp_m_pre = tf.multiply(layer.kernel, tpsp)
            tpsp_m_pre = tf.reduce_sum(tpsp_m_pre,1)
            tpsp_m_pre = tf.divide(tpsp_m_pre,neuron_post.spike_counter_int*conf.n_init_vth)
            tpsp_m_pre = 1.0 + tpsp_m_pre.numpy()

            d_s_w = tpsp

            del_in_e = tf.multiply(del_in, tpsp_m_pre)
            del_in_e = tf.tile(tf.expand_dims(del_in_e,1), [1,tf.shape(layer.kernel)[0],1])
            #d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])
            d_s_w_e = d_s_w

            grad_b = tf.reduce_sum(del_in,0)
            grad = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)

            grads_and_vars.append((grad, layer.kernel))
            grads_and_vars.append((grad_b, layer.bias))

        optimizer.apply_gradients(grads_and_vars)

    return avg_loss.result(), 100*accuracy.result()


def train_snn_one_epoch_mnist(model, optimizer, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')

    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:
        grads_and_vars = []

        with tf.GradientTape(persistent=True) as tape:
            predictions_times = model(images, f_training=True)

            if predictions_times.shape[1] != labels.numpy().shape[0]:
                predictions_times_trimmed = predictions_times[:,0:labels.numpy().shape[0],:]
                d = predictions_times.shape[1] - labels.numpy().shape[0]

                labels_expandded = tf.concat([labels, tf.zeros((d,)+(labels.numpy().shape[1],))],axis=0)
                f_exp = True
            else:
                predictions_times_trimmed = predictions_times
                labels_expandded = labels
                f_exp = False


            predictions_trimmed = predictions_times_trimmed[-1]

            #loss_value = loss(predictions_trimmed, labels)
            #loss_value, delta = loss_cross_entropy(predictions_trimmed, labels)
            loss_value, delta = loss_mse(predictions_trimmed, labels)

            avg_loss(loss_value)

            #accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
            accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


            #predictions = predictions_times[-1]

            #del_last = loss_value - labels_expandded

            #softmax = tf.nn.softmax(predictions_trimmed)

        #del_last = tf.cast(softmax,tf.float32) - tf.cast(labels,tf.float32)
        #del_last = tf.cast(predictions_trimmed,tf.float32) - tf.cast(labels,tf.float32)

        del_last = delta

        if f_exp :
            del_last = tf.concat([del_last, tf.zeros((d,)+(del_last.numpy().shape[1],))],axis=0)

        idx_layer_last = 1

        for idx_layer in range(idx_layer_last,-1,-1):

            layer = model.list_l[idx_layer]
            act = model.list_a[idx_layer]
            neuron = model.list_n[idx_layer]
            neuron_post = model.list_n[idx_layer+1]

            # backprop
            if idx_layer == idx_layer_last:
                del_in = del_last
            else:
                del_in = del_out

            del_out = bp_Dense(layer, del_in)
            del_out = bp_relu(act, del_out)
            #del_out = tf.multiply(del_out,1.0/conf.n_init_vth)

            # weight update
            #d_s_w = act
            #d_s_w = neuron.get_spike_rate()
            #d_s_w = neuron.get_tot_psp()/neuron.get_spike_count_int()
            d_s_w = neuron.get_spike_rate()

            del_in_e = del_in
            del_in_e = tf.tile(tf.expand_dims(del_in_e,1), [1,tf.shape(layer.kernel)[0],1])
            d_s_w_e = tf.tile(tf.expand_dims(d_s_w,2), [1,1,tf.shape(layer.kernel)[1]])

            grad_b = tf.reduce_sum(del_in,0)
            grad = tf.reduce_sum(tf.multiply(del_in_e, d_s_w_e),0)

            grads_and_vars.append((grad, layer.kernel))
            grads_and_vars.append((grad_b, layer.bias))

        optimizer.apply_gradients(grads_and_vars)

    return avg_loss.result(), 100*accuracy.result()


# backprop
def bp_Dense(layer, del_in):
    return tf.matmul(del_in, layer.kernel, transpose_b=True)

def bp_relu(act, del_in):
    return tf.multiply(del_in,act)

def grad_Dense(layer, delta, d_local):
    delta_e = tf.tile(tf.expand_dims(delta,1), [1,tf.shape(layer.kernel)[0],1])
    d_local_e = tf.tile(tf.expand_dims(d_local,2), [1,1,tf.shape(layer.kernel)[1]])

    grad_b = tf.reduce_sum(delta,0)
    grad = tf.reduce_sum(tf.multiply(delta_e, d_local_e),0)

    return grad, grad_b


# temporal coding - trainting time_const, time_delay
def train_time_const_delay_tmeporal_coding(model, dataset, conf):
    tf.train.get_or_create_global_step()
    #avg_loss = tfe.metrics.Mean('loss')
    #accuracy = tfe.metrics.Accuracy('accuracy')

    avg_loss = tf.keras.metrics.Mean('loss')
    accuracy = tf.keras.metrics.Accuracy('accuracy')

    #for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    #for (batch, (images, labels)) in enumerate(tf.Iterator(dataset)):
    for (images, labels) in dataset:

        #with tf.GradientTape(persistent=True) as tape:
        predictions_times = model(images, f_training=True)

        if predictions_times.shape[1] != labels.numpy().shape[0]:
            predictions_times_trimmed = predictions_times[:,0:labels.numpy().shape[0],:]
            d = predictions_times.shape[1] - labels.numpy().shape[0]

        else:
            predictions_times_trimmed = predictions_times


        predictions_trimmed = predictions_times_trimmed[-1]

        loss_value = loss(predictions_trimmed, labels)

        avg_loss(loss_value)

        #accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
        accuracy(tf.argmax(predictions_trimmed,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))


        #predictions = predictions_times[-1]

        softmax = tf.nn.softmax(predictions_trimmed)



    return avg_loss.result(), 100*accuracy.result()



