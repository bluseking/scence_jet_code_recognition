import numpy as np
import sys, os
import time
import os.path
import itertools
import glob

sys.path.append(os.getcwd())


# crnn packages
import torch
#from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
import params
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--images_path', type=str, default='/home/cad488/crnn_chinese_characters_rec/to_lmdb/test_train_images/Test/', help='the path to your images')
parser.add_argument('--images_path', type=str, default='/home/cad488/Test/', help='the path to your images')
opt = parser.parse_args()


# crnn params
# 3p6m_third_ac97p8.pth
#crnn_model_path = '/home/cad488/crnn_scene_recognition_kinds_36/expr/crnn_no_IO_30_56371.pth" 19
crnn_model_path = '/home/cad488/crnn_scene_recognition_kinds_36/expr/rnn_no_IO_100_56371.pth'
alphabet = str1
nclass = len(alphabet)+1


# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')

    ## 
    w = int(image.size[0] / (280 * 1.0 / params.imgW))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    #if torch.cuda.is_available():
        #image = image.cuda()
    image = image.view(1, *image.size())
    #image = Variable(image)

    model.eval()
    preds = model(image)
    print("preds first=", preds.size())
    _, preds = preds.max(2)
    print("preds pre=", preds.size())
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    print("preds size=", preds.size())
    #preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds_size = torch.IntTensor([preds.size(0)])
    #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s => %-20s' % (raw_pred, sim_pred))
    print('results: {0}'.format(sim_pred))


if __name__ == '__main__':

	# crnn network
    SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
    model = crnn.CRNN(32, 1, nclass, 256)
    #if torch.cuda.is_available():
        #model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path,map_location=torch.device('cpu')))
    
    # read an image
    images_files = list(itertools.chain.from_iterable(glob.glob(os.path.join(opt.images_path, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))
    #images_files = (glob.glob(os.path.join(opt.images_path, "*.jpg")))
    #print(images_files)
    images_files.sort()
    #print("images=",images_files)
    #images_files.sort(key =lambda x : int(os.path.basename(x)[:-4]))#get filename and sort by the filename except for the extention
    for file_name in images_files:
        t1 = time.time()
        image = Image.open(file_name)
        print("-------------------------------------")
        print("file_name=",file_name)
        crnn_recognition(image, model)
        t = time.time()-t1
        print("recognition time:", t)
        #finished = time.time()
        #print('elapsed time: {0}'.format(finished-started))
    
