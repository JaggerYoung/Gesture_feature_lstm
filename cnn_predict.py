import find_mxnet
import mxnet as mx
import logging
import time
import cv2
import random
import glob
import numpy as np
import cPickle as p

BATCH_SIZE = 1

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

	self.pad = 0
	self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n,x in zip(self.label_names, self.label)]

def readData(Filename):
    data_1 = []
    data_2 = []
    data_list = []
    f = open(Filename, 'r')
    total = f.readlines()

    for eachLine in range(len(total)):
        pic_x = []
	pic_y = []
	tmp = total[eachLine].split('\n')
	tmp_1, tmp_2, tmp_3 = tmp[0].split(' ',2)
	tmp_1 = '/data/zhigang.yang'+tmp_1
	for filename in glob.glob(tmp_1+'/img*.jpg'):
	    pic_x.append(filename)
	pic_x.sort()
	for i in range(len(pic_x)):
	    data_1.append(pic_x[i])
	    data_2.append(int(tmp_3))
	data_list.append(len(pic_x))
    f.close()
    return (data_1, data_2, data_list)

def readImg(Filename_1, data_shape):
    mat = [] 

    img_1 = cv2.imread(Filename_1, cv2.IMREAD_COLOR)
    r,g,b = cv2.split(img_1)
    r = cv2.resize(r, (data_shape[2], data_shape[1]))
    g = cv2.resize(g, (data_shape[2], data_shape[1]))
    b = cv2.resize(b, (data_shape[2], data_shape[1]))
    r = np.multiply(r, 1/255.0)
    g = np.multiply(g, 1/255.0)
    b = np.multiply(b, 1/255.0)

    mat.append(r)
    mat.append(g)
    mat.append(b)

    return mat

class InceptionIter(mx.io.DataIter):
    def __init__(self, fname, batch_size, data_shape):
        self.batch_size = batch_size
	self.fname = fname
	self.data_shape = data_shape
	(self.data_1, self.data_3, self.data_2) = readData(self.fname)
	self.num = len(self.data_1)/batch_size
	print len(self.data_1)
	
	self.provide_data = [('data', (batch_size,) + data_shape)]
	self.provide_label = [('label', (batch_size, ))]

    def __iter__(self):
        for k in range(self.num):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        idx = k * self.batch_size + i 
		img = readImg(self.data_1[idx], self.data_shape)
		data.append(img)
		label.append(self.data_3)

	    data_all = [mx.nd.array(data)]
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':
    
    batch_size = BATCH_SIZE
    data_shape = (3,299,299)

    train_file = '/data/zhigang.yang/gesture_train.txt'
    test_file = '/data/zhigang.yang/gesture_test.txt'

    data_train = InceptionIter(train_file, batch_size, data_shape)
    data_test = InceptionIter(test_file, batch_size, data_shape)

    print data_train.provide_data, data_train.provide_label

    devs = [mx.context.gpu(2)]
    model = mx.model.FeedForward.load("./googlenet_model/Inception-7", epoch=0001, ctx=devs, num_batch_size=BATCH_SIZE)

    internels = model.symbol.get_internals()
    #print internels.list_outputs()
    fea_symbol = internels['flatten_output']
    feature_exactor = mx.model.FeedForward(ctx=devs, symbol=fea_symbol, num_batch_size=1,
                                           arg_params=model.arg_params, aux_params=model.aux_params,
					   allow_extra_params=True)

    cnn_train_result = feature_exactor.predict(data_train)
    cnn_test_result = feature_exactor.predict(data_test)

    print mx.nd.array(cnn_train_result).shape
    print mx.nd.array(cnn_test_result).shape

    train_data_file = 'train_data.data'
    f_1 = file(train_data_file, 'w')
    p.dump(cnn_train_result, f_1)
    f_1.close()
    test_data_file = 'test_data.data'
    f_2 = file(test_data_file, 'w')
    p.dump(cnn_test_result, f_2)
    f_2.close()

    (tmp_1, train_label, train_list) = readData(train_file)
    (tmp_2, test_label, test_list) = readData(test_file)

    print mx.nd.array(train_label).shape
    print mx.nd.array(test_label).shape

    train_label_file = 'train_label.data'
    f_3 = file(train_label_file, 'w')
    p.dump(train_label, f_3)
    f_3.close()

    test_label_file = 'test_label.data'
    f_4 = file(test_label_file, 'w')
    p.dump(test_label, f_4)
    f_4.close()

    train_list_file = 'train_list.data'
    f_5 = file(train_list_file, 'w')
    p.dump(train_list, f_5)
    f_5.close()

    test_list_file = 'test_list.data'
    f_6 = file(test_list_file, 'w')
    p.dump(test_list, f_6)
    f_6.close()
