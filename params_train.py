import alphabets

random_sample = True
keep_ratio = False
adam = False
adadelta = False
saveInterval = 1 #2
valInterval = 20000
n_test_disp = 10
displayInterval = 5
experiment = './expr'
alphabet = alphabets.alphabet
crnn = './expr/crnn_Rec_done_282_31336.pth'
beta1 =0.5
lr = 0.0001
niter = 1000
nh = 256
imgW = 600
imgH = 64
batchSize = 16
workers = 2
