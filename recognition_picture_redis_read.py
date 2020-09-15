import sys
import os
from random import randint
from collections import OrderedDict
import time 
import multiprocessing
import threading
import zmq
import params
from zmq.utils.monitor import recv_monitor_message
import redis
import json
import datetime
import base64


#print(os.getcwd())
#sys.path.append(os.getcwd())

import torch
from  torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
from io import BytesIO,StringIO

str1 = alphabets.alphabet

HEARTBEAT_LIVENESS = 5
HEARTBEAT_INTERVAL = 1.0
INTERVAL_INIT = 1
INTERVAL_MAX = 32

PPP_READY = b"\x01"

NBR_WORKERS = 1

crnn_model_path = '~/crnn_scene_recognition_kinds_36/expr/crnn_no_IO_4_56371.pth'
FRONTEND_HOST = "tcp://*:5678"
BACKEND_HOST= "tcp://*:5679"

#redis client
redis_client = redis.StrictRedis(host="localhost", port=6379)

alphabet = str1
nclass = len(alphabet)+1
model = None


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
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s => %-20s' % (raw_pred, sim_preda
    print('result:{0}'.format(sim_pred))
    return ('{0}'.format(sim_pred))


def main():
    image_result_tuple = redis_client.zscan("result",0,match="*2019-08-30-22:14*") 
    for image_result in image_result_tuple[1]:
        cached_data_as_dict = json.loads(image_result[0])
        image = base64.b64decode(cached_data_as_dict['image_content'])
        print(image)
        print(cached_data_as_dict['create_time'])
        print(cached_data_as_dict['recognize_result'])
    #image_store = image_content.read()
    #image = Image.open(image_content)
    #insert_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    #result = crnn_recognition(image, model).lstrip('- ABCDEFGHJKLMNPQRSTUVWXYZ')
    #dict_object  = {"cteate_time":insert_time,"image_content":base64.b64encode(image_store).decode("utf8"),"recognize_result":result} 
    #stringified_dict_obj = json.dumps(dict_object)
    #redis_client.zadd("result",{stringified_dict_obj: t1})

if __name__ == '__main__':
    
    # crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    #if torch.cuda.is_available():
        #model = model.cuda()
    model_path = os.path.expanduser(crnn_model_path)
    print('loading pretrained model from {0}'.format(model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    main()
