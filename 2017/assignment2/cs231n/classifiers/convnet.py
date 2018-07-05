import numpy as np

from cs231n.classifiers.fc_net import *
from cs231n.layer_utils import *

class MyConvNet(object):
  def __init__(self, input_dim=(3, 32, 32),
               conv_params=[{'filter_num':16,
                             'filter_size':5,
                             'stride':1}],
               pool_params=[{'pool_height':2,
                             'pool_width':2,
                             'stride':2}], # or None
               hidden_dims=[2730, 2730, 2730], # (2/3 * input) X 3
               num_classes=10, weight_scale=1e-3, reg=0.0,
               dropout=0, use_batchnorm=False, dtype=np.float32, seed=None):

    self.reg = reg
    self.use_batchnorm = use_batchnorm
    self.num_convs = len(conv_params)
    self.conv_params = conv_params
    self.pool_params = pool_params
    self.params = {}
    self.dtype = dtype

    F = 0
    C,H,W = input_dim
    for i in range(len(conv_params)):
      F = conv_params[i]['filter_num']
      HH = conv_params[i]['filter_size']
      WW = conv_params[i]['filter_size']
      stride = conv_params[i]['stride']
      pad = (conv_params[i]['filter_size'] - stride) // 2

      # Init weights
      self.params['conv_W'+str(i+1)]=np.random.randn(F, C, HH, WW) * weight_scale
      self.params['conv_b'+str(i+1)]=np.zeros(F)

      # Additional updates
      self.conv_params[i]['pad'] = pad

      C = F
      H = 1 + (H - HH + 2 * pad) // stride
      W = 1 + (W - WW + 2 * pad) // stride

      # Considering pooling
      if pool_params[i] != None:
        pool_HH = pool_params[i]['pool_height']
        pool_WW = pool_params[i]['pool_width']
        pool_stride = pool_params[i]['stride']

        H = 1 + (H - pool_HH) // pool_stride
        W = 1 + (W - pool_WW) // pool_stride

    fc_input_dim = F * H * W
    self.fc = FullyConnectedNet(hidden_dims=hidden_dims,
                                input_dim=fc_input_dim,
                                num_classes=num_classes,
                                dropout=dropout,
                                weight_scale=weight_scale,
                                use_batchnorm=self.use_batchnorm,
                                reg=self.reg,
                                dtype=self.dtype,
                                seed=seed)
    self.params.update(self.fc.params) # To update all weights and bias of FC

    # Batch Normalization Parameter
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in range(self.num_convs)]
      gammas = {'conv_gamma' + str(i+1): np.ones(conv_params[i]['filter_num'])
                    for i in range(self.num_convs)}
      betas = {'conv_beta' + str(i+1): np.zeros(conv_params[i]['filter_num'])
                    for i in range(self.num_convs)}
      self.params.update(gammas)
      self.params.update(betas)

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Setting mode: test/train
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    # Calculate scores
    scores = None
    cache = {}
    x = X
    for i in range(self.num_convs):
      # Whether to use Batch Normalization
      bn_param = (self.params['conv_gamma' + str(i+1)],\
                  self.params['conv_beta' + str(i+1)],\
                  self.bn_params[i]) \
                    if self.use_batchnorm is True else None

      w = self.params['conv_W' + str(i+1)]
      b = self.params['conv_b' + str(i+1)]

      x, cache['conv' + str(i+1)] = \
          conv_bn_relu_pool_forward(x, w, b,\
                                    conv_param=self.conv_params[i],\
                                    bn_param=bn_param,\
                                    pool_param=self.pool_params[i])

    if mode == 'test':
      scores = self.fc.loss(x, y)
      return scores

    # Calculate loss and gradients
    loss, grads = self.fc.loss(x, y)
    dout = grads['dFC'].reshape(x.shape)
    for i in range(self.num_convs, 0, -1):
      dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dout, cache['conv' + str(i)])
      dout = dx

      grads['conv_W' + str(i)] = dw
      grads['conv_b' + str(i)] = db
      if self.use_batchnorm:
        grads['conv_gamma' + str(i)] = dgamma
        grads['conv_beta' + str(i)] = dbeta

      # Regularization
      loss += 0.5 * self.reg*np.sum(self.params['conv_W' + str(i)]**2)
      grads['conv_W' + str(i)] += self.reg * self.params['conv_W' + str(i)]

    return loss, grads
