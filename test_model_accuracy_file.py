import numpy as np
import sys, os
import time
import os.path
import itertools
import glob

sys.path.append(os.getcwd())


# crnn packages
import torch
from torch.autograd import Variable
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
crnn_model_root = 'crnn_scene_recognition_kinds_36/expr/crnn_no_IO_4_56371.pth'
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
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s => %-20s' % (raw_pred, sim_pred))
    return ('{0}'.format(sim_pred))


def makefile(path):
    test_result_file = os.path.exists(path)
    if (not test_result_file):
        os.mknod(path)


if __name__ == '__main__':

	# crnn network
    SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
         model = model.cuda()
    model_result_name = 'model_result_name.txt'
    home_dir = os.getenv('HOME')
    crnn_model_path = os.path.join(home_dir,crnn_model_root)
    print('loading pretrained model from the path:{0}'.format(crnn_model_path))
   
    model.load_state_dict(torch.load(crnn_model_path,map_location=torch.device('cpu')))
    for image_fold in os.listdir(opt.images_path):
        right = 0
        wrong = 0
        makefile(os.path.join(os.path.join(opt.images_path,image_fold),model_result_name))
        for image_root,subDir,images in os.walk(os.path.join(opt.images_path,image_fold)):
            if images:
                images_files = list(itertools.chain.from_iterable(glob.glob(os.path.join(image_root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))
                for image_file in images_files:
                    image = Image.open(image_file)
                    result = crnn_recognition(image, model).lstrip('- ABCDEFGHJKLMNPQRSTUVWXYZ').rstrip('-')
                    print('results: {0}'.format(result))
                    if result == image_fold:
                        right = right + 1
                    else:
                        wrong = wrong + 1 
        accuracy = right/float(right+wrong)
        with open(os.path.join(os.path.join(opt.images_path,image_fold),model_result_name),'a') as f:
            f.write('  '+ str(accuracy)+' '+ 'right= '+str(right) +' '+'wrong= '+str(wrong)+'\n')                        
