import torch.nn as nn
import torch

#add encode and decode network





class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        #print("input =", input)
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x80 or [64,16,90]
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x40 or [128,8,45] 
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x41  or [128,8,46]
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x42 or [512,2,48]
        convRelu(6, True)  # 512x1x41 or [512,2,47]
        self.cnn = cnn
        #print("cnn in init",cnn)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        #print("self inupt",input.shape)
        #print('---forward propagation---') #input size = torch.Size([16, 1, 32, 160])
        conv = self.cnn(input)

        #print("conv shape = ", conv.shape)
        #print("type conv =",conv.size()) # conv represents a Torch.Tensor
        #print(conv.size()) #[batch_size,512,1,width] = torch.Size([16, 512, 1, 41])
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width or torch.squeeze(conv,2) 2 represents the dim = [batch_size,512,41]
        conv = conv.permute(2, 0, 1)  # [w, b, c], w represent timestep
        #print(conv.size()) # width batch_size channel
        # rnn features
        output = self.rnn(conv)#[w, b, c] ====>[timestep, batch, class]

        output = torch.nn.functional.log_softmax(output, dim=2)
        #print(output.size(0))
        # print(output.size())# width*batch_size*nclass
        #print("output shape = ", output.shape)
        return output
