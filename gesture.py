import os,sys
import random
import find_mxnet
import mxnet as mx
import string
import math

import numpy as np
import cPickle as p
from lstm import lstm_unroll
#from cnn_predict import vgg_predict
#from cnn_predict import get_label

BATCH_SIZE = 15

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.label_names, self.label)]

def Accuracy(label, pred):
    SEQ_LEN = 10
    hit = 0.
    total = 0.
    label = label.T.reshape(-1,1)
    for i in range(BATCH_SIZE*SEQ_LEN):
        maxIdx = np.argmax(pred[i])
	if maxIdx == int(label[i]):
	    hit += 1.0
	total += 1.0
    return hit/total

class LRCNIter(mx.io.DataIter):
    def __init__(self, dataset, labelset, num, listset, batch_size, seq_len, init_states):
        
	self.batch_size = batch_size
	self.count = num/batch_size
	self.seq_len = seq_len
	self.dataset = dataset
	self.labelset = labelset
	self.listset = listset
	
	self.init_states = init_states
	self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

	self.provide_data = [('data',(batch_size, seq_len, 2048))]+init_states
	self.provide_label = [('label',(batch_size, seq_len, ))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
	for k in range(self.count):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        data_seq = []
		label_seq = []
		idx = k * self.batch_size + i
		ret = 0
		for j in range(idx):
		    ret += self.listset[j]
		tmp = random.randint(0, self.listset[idx]-self.seq_len)
		for j in range(self.seq_len):
	            idx_1 = ret + tmp + j
		    data_seq.append(self.dataset[idx_1])
		    label_seq.append(self.labelset[idx_1])
		data.append(data_seq)
		label.append(label_seq)
	
	    data_all = [mx.nd.array(data)]+self.init_state_arrays
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']+init_state_names
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':
    num_hidden = 2048
    num_lstm_layer = 2
    batch_size = BATCH_SIZE

    num_epoch = 500
    learning_rate = 0.0025
    momentum = 0.0015
    num_label = 5
    seq_len = 10
    
    contexts = [mx.context.gpu(3)]

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, num_hidden, num_label)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    f_1 = file('train_data.data')
    x_train = p.load(f_1)
    f_2 = file('test_data.data')
    x_test = p.load(f_2)

    f_3 = file('train_label.data')
    y_train = p.load(f_3)
    f_4 = file('test_label.data')
    y_test = p.load(f_4)

    f_5 = file('train_list.data')
    train_list = p.load(f_5)
    f_6 = file('test_list.data')
    test_list = p.load(f_6)
     
    #print mx.nd.array(x_train).shape, mx.nd.array(x_test).shape
    #print mx.nd.array(x_test).shape, mx.nd.array(y_test).shape

    data_train = LRCNIter(x_train, y_train, len(train_list), train_list, batch_size, seq_len, init_states)
    data_test = LRCNIter(x_test, y_test, len(test_list), test_list, batch_size, seq_len, init_states)
    #print data_train.provide_data, data_train.provide_label

    symbol = sym_gen(seq_len)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
				 num_epoch=num_epoch,
				 learning_rate=learning_rate,
				 momentum=momentum,
				 wd=0.00001,
				 initializer=mx.init.Xavier(factor_type="in",magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'begin fit'
    batch_end_callbacks = [mx.callback.Speedometer(BATCH_SIZE, 100)]
    debug_metrics = mx.metric.np(Accuracy)

    model.fit(X=data_train, eval_data=data_test, eval_metric=debug_metrics, batch_end_callback=batch_end_callbacks)
