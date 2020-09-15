import sys
import os
from random import randint
from collections import OrderedDict
import time 
import multiprocessing
import threading
import zmq
import params
#from zmq.utils.monitor import recv_monitor_message
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

crnn_model_path = '~/crnn_scene_recognition_kinds_36/expr/rnn_no_IO_104_56371.pth'
FRONTEND_HOST = "tcp://*:5678"
BACKEND_HOST= "tcp://*:5679"

#redis client
redis_client = redis.StrictRedis(host="192.168.0.2", port=6379)

alphabet = str1
nclass = len(alphabet)+1
model = None
# crnn文本信息识别
#EVENT_MAP = {}
#print("Event names:")
#for name in dir(zmq):
#    if name.startswith('EVENT_'):
#        value = getattr(zmq, name)
#        print("%21s : %4i" % (name, value))
#        EVENT_MAP[value] = name
#
#
#def event_monitor(monitor):
#    while monitor.poll():
#        evt = recv_monitor_message(monitor)
#        evt.update({'description': EVENT_MAP[evt['event']]})
#        print("Event: {}".format(evt))
#        if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
#            break
#    monitor.close()
#    print()
#    print("event monitor thread done!")

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


class Worker(object):
    def __init__(self, address):
        self.address = address
        self.expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS

class WorkerQueue(object):
    def __init__(self):
        self.queue = OrderedDict()

    def ready(self, worker):
        self.queue.pop(worker.address, None)
        self.queue[worker.address] = worker

    def purge(self):
        """Look for & kill expired workers."""
        t = time.time()
        expired = []
        for address,worker in self.queue.items():
            if t > worker.expiry:  # Worker expired
                expired.append(address)
        for address in expired:
            print ("expired worker: %s" % address)
            self.queue.pop(address, None)

    def next(self):
        address, worker = self.queue.popitem(False)
        return address

def worker_task(worker_url,ident): #receive the request from the client or send the message to backend
    context = zmq.Context(1)
    worker = context.socket(zmq.DEALER)
    worker.identity = u"{}-{}".format(randint(0, 0x10000), randint(0, 0x10000)).encode("ascii")
    worker.connect(worker_url)

    # Tell broker we're ready for work
    worker.send(PPP_READY)
    
    
    poll_worker = zmq.Poller() 
    poll_worker.register(worker, zmq.POLLIN)

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
                print("I: Normal reply")
                t1 = time.time()
                #print("t1=",t1)
                #print("type t1",type(t1))
                #print("frames[2]",frames[2])
                client_key = frames[0]
                image_content = BytesIO(frames[2])
                image_store = image_content.read()
                #print("content",image_store)
                #print("s=",s)
                #print("type Image=",type(image_content))
                #image = Image.open(BytesIO(frames[2]))
                image = Image.open(image_content)
                insert_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                score = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
                result = crnn_recognition(image, model).lstrip('- ABCDEFGHJKLMNPQRSTUVWXYZ')
                dict_object  = {"create_time":insert_time,"image_content":base64.b64encode(image_store).decode("utf8"),"recognize_result":result} 
                stringified_dict_obj = json.dumps(dict_object)
                redis_client.zadd("result",{stringified_dict_obj: score})
                t2 = time.time() - t1
                print("recogition time:", t2)
                print("result=",result)
                #image.save("/home/centos7/test_images/"+ time.asctime(time.localtime(time.time()))+".jpg")
                worker.send_multipart([frames[0], b"", result.encode("ascii")])
            else:
                print ("E: Invalid message: %s" % frames)

def main():
    """Load balancer main loop."""
    # Prepare context and sockets
    url_worker = "tcp://localhost:5679"
    context = zmq.Context(1)
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.ROUTER)
    #front_monitor = frontend.get_monitor_socket()
    #back_monitor = backend.get_monitor_socket()

    frontend.bind(FRONTEND_HOST)
    backend.bind(BACKEND_HOST)
    # Start background tasks
    def start(task, *args):
        process = multiprocessing.Process(target=task, args=args)#多进程，每个进程需要自己的context
        #process = threading.Thread(target=task,args=args) #多线程，参数中的变量每个线程各自拥有
        process.daemon = True
        process.start()
    for i in range(NBR_WORKERS):
        start(worker_task, url_worker,i)
    
    #t = threading.Thread(target=event_monitor, args=(front_monitor,))
    #t.start()
    #t2 = threading.Thread(target=event_monitor, args=(back_monitor,))
    #t2.start()
    #start(event_monitor,front_monitor)
    #start(event_monitor,back_monitor)

    # Initialize main loop state
    workers = WorkerQueue()
    poller = zmq.Poller()
    #Only poll for requests from backend until workers are available
    poll_workers = zmq.Poller()
    poll_workers.register(backend, zmq.POLLIN)

    poll_both = zmq.Poller()
    poll_both.register(frontend, zmq.POLLIN)
    poll_both.register(backend, zmq.POLLIN)

    while True:
        if len(workers.queue) > 0:
            poller = poll_both
        else:
            poller = poll_workers
        sockets = dict(poller.poll(HEARTBEAT_INTERVAL * 1000))
        #print("sockets=:",sockets)
        #print("sockets backend:",sockets.get(backend))
        #print("sockets frontend:",sockets.get(frontend))
        #print(zmq.POLLIN)
        if backend in sockets:
            # Handle worker activity on the backend
            frames = backend.recv_multipart()
            print("get from workers:",frames)
            if not frames:
                break
            address  = frames[0]
            #print("length socks:",len(workers.queue))
            #print("workers queue:",workers.queue)
            #if len(workers.queue) == 0:
               #poller.register(frontend, zmq.POLLIN)
            workers.ready(Worker(address))
            msg = frames[1:]
            if len(msg) == 1:
                if msg[0] not in (PPP_READY):
                    print("E: Invaild message from worker: %s" %msg)
            else:
                frontend.send_multipart(msg)

        if frontend in sockets:
            frames = frontend.recv_multipart()
            #print("get from clients",frames[0])
            if not frames:
                break
            frames.insert(0,workers.next())
            #frames = [workes.next, ''] + frames
            backend.send_multipart(frames)
            #if len(workers.queue) == 0:
                #poller.unregister(frontend)
        
        #workers.purge()
    
    # Clean up
    backend.close()
    frontend.close()
    context.term()


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
