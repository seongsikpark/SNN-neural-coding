from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe

import math

import train
from tqdm import tqdm

import pandas as pd

#
def test(model, dataset, num_dataset, conf, f_val=False, epoch=0, f_val_snn=False):
    avg_loss = tf.keras.metrics.Mean('loss')

    if conf.nn_mode=='SNN' or f_val_snn:
        accuracy_times = []
        accuracy_result = []

        accuracy_time_point = list(range(conf.time_step_save_interval,conf.time_step,conf.time_step_save_interval))
        accuracy_time_point.append(conf.time_step)
        argmax_axis_predictions=1

        num_accuracy_time_point=len(accuracy_time_point)

        if f_val==False:
            print('accuracy_time_point')
            print(accuracy_time_point)

            print('num_accuracy_time_point: {:d}'.format(model.num_accuracy_time_point))

        for i in range(num_accuracy_time_point):
            accuracy_times.append(tf.keras.metrics.Accuracy('accuracy'))

        num_batch=int(math.ceil(float(num_dataset)/float(conf.batch_size)))

        print_loss = True
        if conf.f_train_time_const and print_loss:
            list_loss_prec = list(range(num_batch))
            list_loss_min = list(range(num_batch))
            list_loss_max = list(range(num_batch))

            list_tc = list(range(num_batch))
            list_td = list(range(num_batch))

        if f_val==False:
            pbar = tqdm(range(1,num_batch+1),ncols=80)
            pbar.set_description("batch")

        if f_val_snn:
            model.f_done_preproc = False

        idx_batch=0
        for (images, labels_one_hot) in dataset:
            labels = tf.argmax(labels_one_hot,axis=1,output_type=tf.int32)

            f_resize_output = False
            if conf.batch_size != labels.shape:
                concat_dim = conf.batch_size-labels.numpy().shape[0]
                f_resize_output = True

                labels = tf.concat([labels,tf.zeros(shape=[concat_dim],dtype=tf.int32)],0)
                images = tf.concat([images,tf.zeros(shape=(concat_dim,)+tuple(images.shape[1:]),dtype=images.dtype)],0)


            if idx_batch!=-1:

                # predictions_times - [saved time step, batch, output dim]
                predictions_times = model(images, f_training=False, f_val_snn=f_val_snn, epoch=epoch)

                if f_resize_output:
                    labels = labels[0:conf.batch_size-concat_dim]
                    predictions_times = predictions_times[:,0:conf.batch_size-concat_dim,:]

                if predictions_times.shape[1] != labels.numpy().shape[0]:
                    predictions_times = predictions_times[:,0:labels.numpy().shape[0],:]

                tf.reshape(predictions_times,(-1,)+labels.numpy().shape)

                if f_val:
                    predictions = predictions_times[-1]
                    accuracy = accuracy_times[-1]
                    accuracy(tf.argmax(predictions,axis=argmax_axis_predictions,output_type=tf.int32), labels)

                else:
                    for i in range(num_accuracy_time_point):
                        predictions=predictions_times[i]
                        accuracy = accuracy_times[i]
                        accuracy(tf.argmax(predictions,axis=argmax_axis_predictions,output_type=tf.int32), labels)

                predictions = predictions_times[-1]
                avg_loss(train.loss_cross_entoropy(predictions,labels_one_hot))

                if conf.f_train_time_const and print_loss:
                    [loss_prec, loss_min, loss_max] = model.get_time_const_train_loss()

                    list_loss_prec[idx_batch]=loss_prec.numpy()
                    list_loss_min[idx_batch]=loss_min.numpy()
                    list_loss_max[idx_batch]=loss_max.numpy()



            if f_val==False:
                pbar.update()

            if conf.f_train_time_const:
                print("idx_batch: {:d}".format(idx_batch))
                num_data=(idx_batch+1)*conf.batch_size+conf.time_const_num_trained_data+(epoch)*conf.num_test_dataset

                print("num_data: {:d}".format(num_data))
                if num_data%conf.time_const_save_interval==0:
                    fname = conf.time_const_init_file_name + '/' + conf.model_name
                    fname+="/tc-{:d}_tw-{:d}_itr-{:d}".format(conf.tc,conf.time_window,num_data)

                    if conf.f_train_time_const_outlier:
                        fname+="_outlier"

                    print("save time constant: file_name: {:s}".format(fname))
                    f = open(fname,'w')

                    # time const
                    for name_neuron, neuron in model.list_neuron.items():
                        if not ('fc3' in name_neuron):
                            f.write("tc,"+name_neuron+","+str(neuron.time_const_fire.numpy())+"\n")
                    f.write("\n")

                    # time delay
                    for name_neuron, neuron in model.list_neuron.items():
                        if not ('fc3' in name_neuron):
                            f.write("td,"+name_neuron+","+str(neuron.time_delay_fire.numpy())+"\n")
                    f.close()
            idx_batch += 1

        #
        if f_val_snn:
            assert False
            model.defused_bn()

        if f_val == False:
            for i in range(num_accuracy_time_point):
                accuracy_result.append(accuracy_times[i].result().numpy())
            print('')
            ret_accu = 100*accuracy_result[-1]
        else:
            ret_accu = 100*accuracy_times[-1].result().numpy()

        if f_val == False:
            pd.set_option('display.float_format','{:.4g}'.format)

            #
            df=pd.DataFrame({'time step': model.accuracy_time_point, 'accuracy': accuracy_result, 'spike count': model.total_spike_count_int[:,-1]/num_dataset, 'spike_count_c1':model.total_spike_count_int[:,0]/num_dataset, 'spike_count_c2':model.total_spike_count_int[:,1]/num_dataset})
            df.set_index('time step', inplace=True)
            print(df)

            if conf.f_save_result:
                # ts: time step
                # tssi: time step save interval
                #f_name_result = conf.path_result_root+'/'+conf.date+'_'+conf.neural_coding
                #f_name_result = conf.path_result_root+'/'+conf.input_spike_mode+conf.neural_coding+'_ts-'+str(conf.time_step)+'_tssi-'+str(conf.time_step_save_interval)
                f_name_result = '{}/{}_{}_ts-{}_tssi-{}_vth-{}'.format(conf.path_result_root,conf.input_spike_mode,conf.neural_coding,str(conf.time_step),str(conf.time_step_save_interval),conf.n_init_vth)

                if conf.neural_coding=="TEMPORAL":
                    f_name_result += outfile_name_temporal(conf)

                f_name_result += '.xlsx'

                df.to_excel(f_name_result)
                print("output file: "+f_name_result)


            if conf.f_train_time_const and print_loss:
                df=pd.DataFrame({'loss_prec': list_loss_prec, 'loss_min': list_loss_min, 'loss_max': list_loss_max})
                fname="./time-const-train-loss_b-"+str(conf.batch_size)+"_d-"+str(conf.num_test_dataset)+"_tc-"+str(conf.tc)+"_tw-"+str(conf.time_window)+".xlsx"
                df.to_excel(fname)


            print('f write date: '+conf.date)

        if conf.verbose_visual:
            model.figure_hold()

    else:
        accuracy=tf.metrics.Accuracy('accuracy')

        if f_val==False:
            num_batch=int(math.ceil(float(conf.num_test_dataset)/float(conf.batch_size)))
            pbar = tqdm(range(1,num_batch+1),ncols=80)
            pbar.set_description("batch")

        idx_batch = 0
        for (images, labels_one_hot) in dataset:
            if idx_batch!=-1:
                predictions = model(images, f_training=False, epoch=epoch)
                accuracy(tf.argmax(predictions,axis=1,output_type=tf.int32), tf.argmax(labels_one_hot,axis=1,output_type=tf.int32))
                avg_loss(train.loss_cross_entoropy(predictions,labels_one_hot))

            if f_val==False:
                pbar.update()

            idx_batch += 1

        ret_accu = 100*accuracy.result()

        if conf.f_write_stat:
            model.save_activation()

    return avg_loss.result(), ret_accu, 0.0



############################################
# output file name
############################################

def outfile_name_temporal(conf):

    f_name_result = '_tc-'+str(conf.tc)+'_tw-'+str(conf.time_window)+'_tfs-'+str(int(conf.time_fire_start)) \
                    +'_tfd-'+str(int(conf.time_fire_duration))

    if conf.f_load_time_const:
        if conf.f_train_time_const:
            f_name_result += '_trained_data-'+str(conf.time_const_num_trained_data+conf.num_test_dataset)
        else:
            f_name_result += '_trained_data-'+str(conf.time_const_num_trained_data)

    if conf.f_train_time_const:
        if conf.f_train_time_const_outlier:
            f_name_result += '_outlier'

        f_name_result += '_train-tc'

    return f_name_result

