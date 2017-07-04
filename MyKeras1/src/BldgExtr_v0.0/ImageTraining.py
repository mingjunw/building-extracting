"""
Author: Mingjun Wang
Description: ConvNet for building extraction
"""
import BldgExtr

def Train_leNet(args, nkerns=[50, 70, 100, 150, 100, 70, 70], batch_size=1):

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
    
