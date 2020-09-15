from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn
import re
import params

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

opt = parser.parse_args()
print(opt)

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        #print("cpu_images size =",cpu_images.size())
        #print("cpu_texts size=",cpu_texts.size())
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)#t represents each character index, l represents each word index(each word length)
        print("each character index=",t.size())
        print("each word length or index=",l.size())
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds = crnn(image)
        #print("preds size  ",preds.size())
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        list_1 = []
        for i in cpu_texts:
            list_1.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1

    
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(max_iter*params.batchSize)
    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    print("batch_size train",cpu_images.size())
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def training():
    for total_steps in range(params.niter):
        train_iter = iter(train_loader)
        i = 0
        print(len(train_loader))
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            cost = trainBatch(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (total_steps, params.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()
            if i % params.valInterval == 0:
                val(crnn, test_dataset, criterion)
        if (total_steps+1) % params.saveInterval == 0:
            torch.save(crnn.state_dict(), '{0}/Test_crnn_no_IO_{1}_{2}.pth'.format(params.experiment, total_steps, i))

if __name__ == '__main__':

    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    
    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')

    # read train set
    train_dataset = dataset.lmdbDataset(root=opt.trainroot)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*180
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))#deal with the input images with the different size,and the images were normalized

    # read test set
    # images will be resize to 32*180
    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, transform=dataset.resizeNormalize((params.imgW, params.imgH)))#convert test images into normalize,the images were not tested with batchsize

    nclass = len(params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(params.alphabet)
    criterion = CTCLoss()

    # cnn and rnn,define image, text, length placehoder 
    image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
    text = torch.IntTensor(params.batchSize * 5) # text length no less than batchsize * 5
    length = torch.IntTensor(params.batchSize) # define each word length 

    crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
    if opt.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)# self-define weight initialize function
    print("crnn =",crnn)
    print("crnn parameters =",crnn.cnn)
    if params.crnn != '':
        print('loading pretrained model from %s' % params.crnn)
        crnn.load_state_dict(torch.load(params.crnn))
    
    #for para in crnn.parameters():
        #print("parameters =", para)#para.requires_grad=False 
    #crnn.state_dict()#get parameters list-------->cnn.conv0.weight cnn.conv0.bias cnn.conv1.weight cnn.conv1.bias cnn.conv2.weight cnn.conv2.bias cnn.batchnorm2.weight  cnn.batchnorm2.bias
    #crnn.state_dict().items()#get parameters name and its value ------------>('rnn.1.embedding.bias', tensor([-0.0602, -0.3962, -0.3687, -0.3052, -0.2965, -0.3442, -0.4302, -0.3631,\
        #-0.3303, -0.2937, -0.2485, -0.4897, -0.2815, -0.3473, -0.3228, -0.2575,\
        #-0.3200, -0.3391, -0.4191, -0.2042, -0.5009, -0.4935, -0.3103, -0.2821, \
        #-0.3521, -0.2895, -0.3934, -0.2745, -0.3072, -0.2851, -0.2467, -0.3485,\
        #-0.2747, -0.2944, -0.3731, -0.4065, -0.3084, -0.3154, -0.4246],\
       #device='cuda:0'))
    for x in crnn.state_dict():
        print("key value=",x)


    # freezing the convolution layer parameters
    #for param in crnn.cnn.parameters():
        #param.requires_grad = False
    
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer 
    #change optimizer parameters crnn.rnn.parameters
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    training()
