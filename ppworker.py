import sys,os
import os.path
import itertools
from collections import OrderedDict
import time 
import zmq
import params
from random import randint
import redis
import json
import base64
import datetime

sys.path.append(os.getcwd())

import torch
#from  torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
from io import BytesIO

str1 = alphabets.alphabet

HEARTBEAT_LIVENESS = 5
HEARTBEAT_INTERVAL = 1.0
INTERVAL_INIT = 1
INTERVAL_MAX = 32

PPP_READY = b"\x01"

crnn_model_path = '/home/cad488/crnn_scene_recognition_kinds_36/expr/rnn_no_IO_7_37000_0.993125.pth'
BACKEND_HOST= "tcp://localhost:5679"

#redis client
redis_client = redis.StrictRedis(host="192.168.0.2", port=6379)

alphabet = str1
nclass = len(alphabet)+1
model = None
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

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = torch.IntTensor([preds.size(0)])
    #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s => %-20s' % (raw_pred, sim_preda
    print('result:{0}'.format(sim_pred))
    return ('{0}'.format(sim_pred))

def worker_socket(): #receive the request from the client or send the message to backend
    worker_context = zmq.Context(1)
    worker = worker_context.socket(zmq.DEALER)
    worker.identity = u"{}-{}".format(randint(0, 0x10000), randint(0, 0x10000)).encode("ascii")
    poll_worker = zmq.Poller()
    poll_worker.register(worker, zmq.POLLIN)
    worker.connect(BACKEND_HOST)
    worker.send(PPP_READY)#tell the broker queue I'm ready
    
    while True:
        socks = dict(poll_worker.poll(HEARTBEAT_INTERVAL * 1000))  
        #frames = worker.recv_multipart()               # format:[b'client',b'',message_body]
        print("{}".format(worker.identity.decode("ascii")))
         
        if socks.get(worker) == zmq.POLLIN:
            frames = worker.recv_multipart()
            #print("worker gets:",frames)
            if not frames:
                break # Interrupted

            if len(frames) == 3:
                print ("I: Normal reply")
                t1 = time.time()
                client_key = frames[0]
                image_content = BytesIO(frames[2])
                image_store = image_content.read()
                image = Image.open(image_content)
                insert_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                score = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
                result = crnn_recognition(image, model).lstrip('- ABCDEFGHJKLMNPQRSTUVWXYZ')
                dict_object = {"create_time":insert_time,"image_content":base64.b64encode(image_store).decode("utf8"), "recognize_result":result}
                stringified_dict_obj = json.dumps(dict_object)
                redis_client.zadd("result",{stringified_dict_obj: score})
                t2 = time.time()-t1
                print("recogition time:", t2)
                image.save("/home/cad488/test_images/"+ time.asctime(time.localtime(time.time()))+".jpg")
                worker.send_multipart([client_key, b"", result.encode("ascii")])
            else:
                print ("E: Invalid message: %s" % frames)

if __name__ == '__main__':
    
    # crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    #if torch.cuda.is_available():
        #model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path,map_location=torch.device('cpu')))
    worker_socket()
