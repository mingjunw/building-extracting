"""
Author: Mingjun Wang
Description: ConvNet for building extraction
"""
from BldgExtr import *
import math
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from theano.tensor.signal import conv as sgnconv

def Train_leNet(nkerns=[50, 70, 100, 150, 100, 70, 70], n_epochs=20, batch_size=5,learning_rate=0.1):
    imgN = 40
    rng = numpy.random.RandomState(23455)
    #imgobj = Image.open(args.imageName)
    
    theano.config.compute_test_value = 'warn' 
    
    train_set_x = numpy.random.randn(imgN,3,500,500).reshape(imgN,3*500*500)
    train_set_x = theano.tensor._shared(
        numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
    train_set_y = numpy.random.randn(imgN,1,500,500).reshape(imgN,1*500*500)
    train_set_y = theano.tensor._shared(
        numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True)
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    #cols, rows = imgobj.size
    cols, rows = 500,500
    Width = 500 #numpy.int(math.floor(cols / 16.) * 16)
    Height = 500 #numpy.int(math.floor(rows / 16.) * 16)
    

    imgshp0 = (Height, Width)
    imgshp1 = (imgshp0[0]/2, imgshp0[1]/2)
    imgshp2 = (imgshp1[0]/2, imgshp1[1]/2)
    imgshp3 = (imgshp2[0]/2, imgshp2[1]/2)
    imgshp4 = (imgshp3[0]/2, imgshp3[1]/2)
    imgshp5 = (imgshp4[0]/2, imgshp4[1]/2)
    imgshp6 = imgshp5

    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')
    #y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    print '... building the model'

    layer0_input = x.reshape((batch_size, 3, Height, Width)) #.dimshuffle(0, 3, 1, 2)

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imgshp0[0], imgshp0[1]),
        filter_shape=(nkerns[0], 3, 5, 5),    #nkerns=[50, 70, 100, 150, 100, 70, 70]
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imgshp1[0], imgshp1[1]),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),   #nkerns=[50, 70, 100, 150, 100, 70, 70]
        poolsize=(2, 2)
    )

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], imgshp2[0], imgshp2[1]),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),  #nkerns=[50, 70, 100, 150, 100, 70, 70]
        poolsize=(2, 2)
    )

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], imgshp3[0], imgshp3[1]),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),  #nkerns=[50, 70, 100, 150, 100, 70, 70]
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
        poolsize=(1, 1)   #nkerns=[50, 70, 100, 150, 100, 70, 70]
    )

    softmax_input1 = layer_fn.output.dimshuffle(0, 2, 3, 1)
    p_y_given_x1 = T.nnet.softmax(softmax_input1.reshape((batch_size*imgshp1[0]*imgshp1[1], 128)))
    p1 = p_y_given_x1 * T.arange(128).astype('float32')
    net_output = T.sum(p1,axis=1).reshape((batch_size,1,imgshp1[0],imgshp1[1]))/127.

    net_output_2x = UpSampling2x(net_output.reshape((batch_size,1,imgshp1[0],imgshp1[1])),
                                 (batch_size,1,imgshp0[0],imgshp0[1]))
        
    # the setting of train model
    
    #cost = negative_log_likelihood(net_output_2x,y)
    params = layer_fn.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    
    p_1 = 1 / (1 + T.exp(-T.dot(net_output_2x, params[0]) - params[1]))   # Probability that target = 1
    prediction = p_1 > 0.5                    # The prediction thresholded
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (params[0] ** 2).sum()# T
    
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    # Train model
    train_model = theano.function(
        [index],
        cost,
        updates = updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]}
    )
    
    
    
    ###############
    # TRAIN MODEL #
    ###############
    print('... training model')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    
    start_time = time.clock()
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 10 == 0:
                print('training @ iter = ', iter)
            tmpImg = train_model(minibatch_index)[0]

    end_time = time.clock()
    print('training end.....')
    
    #tmpImg = test_model(0)[0] 
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

    
    print
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    
if __name__ == '__main__':
    Train_leNet()