"""
Author: Jiangye Yuan
        Oak Ridge National Laboratory
Description: ConvNet for building extraction
"""

import os
import sys
import time
import argparse

import numpy
import math

import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from theano.tensor.signal import conv as sgnconv

import cPickle
from PIL import Image

imgN = 1
paraFile = 'parafile'
Thr = 127/255. # threshold on distance

def fnParseArgs():

    usage = "usage: python %prog [options]"
    parser = argparse.ArgumentParser(usage)

    parser.add_argument("-q",
                        action="store_false",
                        dest="verbose",
                        help="Quiet",
                        default=False)

    parser.add_argument("-v",
                        action  = "store_true",
                        dest    = "verbose",
                        help    = "Verbose",
                        default = True)

    parser.add_argument("-f",
                        action  = "store",
                        dest    = "imageName",
                        help    = "input image",
                        type    = str,
                        required = True)

    parser.add_argument("-o",
                        action  = "store",
                        dest    = "outputName",
                        help    = "output file name",
                        type    = str,
                        required = True)

    args = parser.parse_args()

    if args.verbose:
        print ('User provided inputs:')
        print ('> IMAGENAME\t\t: %s' % args.imageName)
        print ('> OUTPUTNAME\t\t: %s' % args.outputName)

    return args

def UpSampling2x(input, img_shp):
    filter1 = theano.shared(numpy.asarray([.5, 1, .5], dtype=theano.config.floatX), borrow=True)
    input1fs_0 = theano.shared(numpy.zeros((img_shp[0],img_shp[1],img_shp[2],img_shp[3]), dtype=theano.config.floatX),
                               borrow=True)
    input1fs_0 = T.set_subtensor(input1fs_0[:,:,0:img_shp[2]:2,0:img_shp[3]:2],input)

    img_shp1N = (img_shp[0]*img_shp[1],img_shp[2],img_shp[3])
    input1fs_v = sgnconv.conv2d(input1fs_0.reshape(img_shp1N),
                           filter1.reshape((1,3)),
                           border_mode='full',
                           image_shape=img_shp1N,
                           filter_shape=(1, 1, 3))[:,:,1:-1]
    input1fs = sgnconv.conv2d(input1fs_v,
                           filter1.reshape((3,1)),
                           border_mode='full',
                           image_shape=img_shp1N,
                           filter_shape=(1, 3, 1))[:,1:-1,:].reshape((img_shp[0],img_shp[1],img_shp[2],img_shp[3]))
    return input1fs

def UpSampling4x(input, img_shp):
    filter2 = theano.shared(numpy.asarray([.25, .5, .75, 1, .75, .5, .25],
                                          dtype=theano.config.floatX), borrow=True)
    input2fs_0 = theano.shared(numpy.zeros((img_shp[0],img_shp[1],img_shp[2],img_shp[3]), dtype=theano.config.floatX),
                               borrow=True)
    input2fs_0 = T.set_subtensor(input2fs_0[:,:,2:img_shp[2]:4,2:img_shp[3]:4],input)

    img_shp2N = (img_shp[0]*img_shp[1],img_shp[2],img_shp[3])
    input2fs_v = sgnconv.conv2d(input2fs_0.reshape(img_shp2N),
                           filter2.reshape((1,7)),
                           border_mode='full',
                           image_shape=img_shp2N,
                           filter_shape=(1, 1, 7))[:,:,3:-3]
    input2fs = sgnconv.conv2d(input2fs_v,
                           filter2.reshape((7,1)),
                           border_mode='full',
                           image_shape=img_shp2N,
                           filter_shape=(1, 7, 1))[:,3:-3,:].reshape((img_shp[0],img_shp[1],img_shp[2],img_shp[3]))
    return input2fs

def UpSampling8x(input, img_shp):
    filter3 = theano.shared(numpy.asarray([.125, .25, .375, .5, .625, .75, .875, 1,
                                           .875, .75, .625, .5, .375, .25, .125],
                                          dtype=theano.config.floatX), borrow=True)
    input3fs_0 = theano.shared(numpy.zeros((img_shp[0],img_shp[1],img_shp[2],img_shp[3]), dtype=theano.config.floatX),
                               borrow=True)
    input3fs_0 = T.set_subtensor(input3fs_0[:,:,3:img_shp[2]:8,3:img_shp[3]:8],input)

    img_shp3N = (img_shp[0]*img_shp[1],img_shp[2],img_shp[3])
    input3fs_v = sgnconv.conv2d(input3fs_0.reshape(img_shp3N),
                           filter3.reshape((1,15)),
                           border_mode='full',
                           image_shape=img_shp3N,
                           filter_shape=(1, 1, 15))[:,:,7:-7]
    input3fs = sgnconv.conv2d(input3fs_v,
                           filter3.reshape((15,1)),
                           border_mode='full',
                           image_shape=img_shp3N,
                           filter_shape=(1, 15, 1))[:,7:-7,:].reshape((img_shp[0],img_shp[1],img_shp[2],img_shp[3]))
    return input3fs

def UpSampling16x(input, img_shp):
    a = numpy.linspace(0,1,num=17)
    b = numpy.linspace(1,0,num=17)
    c = numpy.concatenate((a[1:],b[1:-1]))
    filter4 = theano.shared(numpy.asarray(c, dtype=theano.config.floatX), borrow=True)
    input4fs_0 = theano.shared(numpy.zeros((img_shp[0],img_shp[1],img_shp[2],img_shp[3]), dtype=theano.config.floatX),
                               borrow=True)
    input4fs_0 = T.set_subtensor(input4fs_0[:,:,9:(img_shp[2]-0):16,9:(img_shp[3]-0):16],input)

    img_shp3N = (img_shp[0]*img_shp[1],img_shp[2],img_shp[3])
    input4fs_v = sgnconv.conv2d(input4fs_0.reshape(img_shp3N),
                           filter4.reshape((1,31)),
                           border_mode='full',
                           image_shape=img_shp3N,
                           filter_shape=(1, 1, 31))[:,:,15:-15]
    input4fs = sgnconv.conv2d(input4fs_v,
                           filter4.reshape((31,1)),
                           border_mode='full',
                           image_shape=img_shp3N,
                           filter_shape=(1, 31, 1))[:,15:-15,:].reshape((img_shp[0],img_shp[1],img_shp[2],img_shp[3]))
    return input4fs


def relu(x):
    return T.switch(x<0, 0, x)

class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        W_bound = numpy.sqrt(1. / filter_shape[0])
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        if filter_shape[2] == 1 and filter_shape[3] == 1:
            conv_out = conv.conv2d(
                input=input,
                filters=self.W,
                border_mode='full',
                filter_shape=filter_shape,
                image_shape=image_shape
            )
        else:
            fh = (filter_shape[2]-1)/2
            fw = (filter_shape[3]-1)/2

            conv_out = conv.conv2d(
                input=input,
                filters=self.W,
                border_mode='full',
                filter_shape=filter_shape,
                image_shape=image_shape
            )[:,:,fh:-fh,fw:-fw]

        #pooled_out = downsample.max_pool_2d(
        #    input=conv_out,
        #    ds=poolsize,
        #    ignore_border=True
        #)

        pooled_out = pool.pool_2d(
            input=conv_out,
            ws=poolsize,
            ignore_border=True,
            mode='max'
        )

        self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]


def evaluate_lenet(args, nkerns=[50, 70, 100, 150, 100, 70, 70], batch_size=1):

    rng = numpy.random.RandomState(23455)

    imgobj = Image.open(args.imageName)
    cols, rows = imgobj.size
    Width = numpy.int(math.floor(cols / 16.) * 16)
    Height = numpy.int(math.floor(rows / 16.) * 16)
    x1 = numpy.zeros((imgN,Width*Height*3),dtype='float32')
    img = numpy.asarray(imgobj, dtype='float32')
    imgtmp = img[0:Height,0:Width,0:3]/255.
    x1[0,:] = imgtmp.reshape(Height*Width*3)

    test_set_x = theano.tensor._shared(
        numpy.asarray(x1,dtype=theano.config.floatX),borrow=True)

    imgshp0 = (Height, Width)
    imgshp1 = (imgshp0[0]/2, imgshp0[1]/2)
    imgshp2 = (imgshp1[0]/2, imgshp1[1]/2)
    imgshp3 = (imgshp2[0]/2, imgshp2[1]/2)
    imgshp4 = (imgshp3[0]/2, imgshp3[1]/2)
    imgshp5 = (imgshp4[0]/2, imgshp4[1]/2)
    imgshp6 = imgshp5

    index = T.lscalar()
    x = T.matrix('x')

    print '... building the model'

    layer0_input = x.reshape((batch_size, Height, Width, 3)).dimshuffle(0, 3, 1, 2)

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imgshp0[0], imgshp0[1]),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imgshp1[0], imgshp1[1]),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], imgshp2[0], imgshp2[1]),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2)
    )

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], imgshp3[0], imgshp3[1]),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(2, 2)
    )

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], imgshp4[0], imgshp4[1]),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(2, 2)
    )

    layer5 = LeNetConvPoolLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, nkerns[4], imgshp5[0], imgshp5[1]),
        filter_shape=(nkerns[5], nkerns[4], 3, 3),
        poolsize=(1, 1)
    )

    layer6 = LeNetConvPoolLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, nkerns[5], imgshp6[0], imgshp6[1]),
        filter_shape=(nkerns[6], nkerns[5], 3, 3),
        poolsize=(1, 1)
    )
    layer1output_2x = UpSampling2x(layer1.output, (batch_size,nkerns[1],imgshp1[0], imgshp1[1]))
    layer2output_4x = UpSampling4x(layer2.output, (batch_size,nkerns[2],imgshp1[0], imgshp1[1]))
    layer3output_8x = UpSampling8x(layer3.output, (batch_size,nkerns[3],imgshp1[0], imgshp1[1]))
    layer6output_16x = UpSampling16x(layer6.output, (batch_size,nkerns[6],imgshp1[0], imgshp1[1]))


    output_all = T.concatenate([layer0.output,
                                layer1output_2x,
                                layer2output_4x,
                                layer3output_8x,
                                layer6output_16x], axis=1)

    layer_fn = LeNetConvPoolLayer(
        rng,
        input=output_all,
        image_shape=(batch_size, nkerns[0]+nkerns[1]+nkerns[2]+nkerns[3]+nkerns[6], imgshp1[0], imgshp1[1]),
        filter_shape=(128, nkerns[0]+nkerns[1]+nkerns[2]+nkerns[3]+nkerns[6], 1, 1),
        poolsize=(1, 1)
    )

    softmax_input1 = layer_fn.output.dimshuffle(0, 2, 3, 1)
    p_y_given_x1 = T.nnet.softmax(softmax_input1.reshape((batch_size*imgshp1[0]*imgshp1[1], 128)))
    p1 = p_y_given_x1 * T.arange(128).astype('float32')
    net_output = T.sum(p1,axis=1).reshape((batch_size,1,imgshp1[0],imgshp1[1]))/127.

    net_output_2x = UpSampling2x(net_output.reshape((batch_size,1,imgshp1[0],imgshp1[1])),
                                 (batch_size,1,imgshp0[0],imgshp0[1]))

    test_model = theano.function(
        [index],
        [net_output_2x],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    save_file = open(paraFile, 'rb')
    layer0.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer0.params[1].set_value(cPickle.load(save_file), borrow=True)
    layer1.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer1.params[1].set_value(cPickle.load(save_file), borrow=True)
    layer2.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer2.params[1].set_value(cPickle.load(save_file), borrow=True)
    layer3.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer3.params[1].set_value(cPickle.load(save_file), borrow=True)
    layer4.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer4.params[1].set_value(cPickle.load(save_file), borrow=True)
    layer5.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer5.params[1].set_value(cPickle.load(save_file), borrow=True)
    layer6.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer6.params[1].set_value(cPickle.load(save_file), borrow=True)
    layer_fn.params[0].set_value(cPickle.load(save_file), borrow=True)
    layer_fn.params[1].set_value(cPickle.load(save_file), borrow=True)
    save_file.close()

    print 'perform testing'

    start_time = time.clock()

    tmpImg = test_model(0)[0]
    tmpImg = tmpImg.reshape((Height,Width))

    img = x1.reshape((Height,Width,3)) * 255

    img[tmpImg >= 128/255.,0] = 255
    img[(tmpImg > 127/255.) & (tmpImg < 129/255.),2] = 255
    img[(tmpImg > 127/255.) & (tmpImg < 129/255.),1] = 0
    img[(tmpImg > 127/255.) & (tmpImg < 129/255.),0] = 0

    img0 = Image.fromarray(img.astype(numpy.uint8))
    imgfn = args.outputName
    img0.save(imgfn)
    img0 = Image.fromarray((tmpImg * 255).astype(numpy.uint8))
    imgfn = 'maskout.png'
    img0.save(imgfn)

    end_time = time.clock()
    print
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    args = fnParseArgs()
    evaluate_lenet(args)

