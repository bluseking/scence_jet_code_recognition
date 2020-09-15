import sys
sys.path.insert(0, '.')

get_ipython().run_line_magic('matplotlib', 'inline')
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd 
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
from time import time

gb.set_figsize()
img = image.imread('/home/cad488/recognitioned_images_new/110160.jpg')
gb.plt.imshow(img.asnumpy())


#def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
#    Y = [aug(img) for _ in range(num_rows * num_cols)]
#    gb.show_images(Y, num_rows, num_cols, scale)


#apply(img, gdata.vision.transforms.RandomFlipLeftRight())



