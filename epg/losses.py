import chainer as C
import chainer.functions as F
import chainer.links as L
import numpy as np


class Loss(C.Chain):
    """This is the loss optimized by the EPG inner loop agent.
    """

    def set_params_1d(self, params):
        """Set params for ES (theta)
        """
        n = sorted([p for p, _ in self.namedparams()])
        _np = dict(self.namedparams())
        idx = 0
        for e in n:
            _np[e].data[...] = params[idx:idx + _np[e].size].reshape(_np[e].shape)
            idx += _np[e].size

    def get_params_1d(self):
        """Get params for ES (theta)
        """
        n = sorted([p for p, _ in self.namedparams()])
        _np = dict(self.namedparams())
        _np = [_np[e].data.flatten() for e in n]
        return np.concatenate(_np)

    def process_trajectory(self, l):
        raise NotImplementedError

    def loss(self, l):
        raise NotImplementedError


class Conv1DLoss(Loss):
    """Convolutional 1D loss: temporal conv, features are channels.
    """

    def __init__(self, traj_dim_in):
        chan_traj_c0_c1 = 16
        chan_traj_c1_d0 = 32
        units_traj_d0_d1 = 32
        units_traj_d1_d2 = 16

        # This means, 1 input dimension (so we convolve along the temporal axis) and treat
        # each feature dimension as a channel. The temporal axis is always the same length
        # since this is fixed with a buffer that keeps track of the latest data.
        traj_c0 = L.ConvolutionND(
            ndim=1, in_channels=traj_dim_in, out_channels=chan_traj_c0_c1, ksize=6, stride=5)
        traj_c1 = L.ConvolutionND(
            ndim=1, in_channels=chan_traj_c0_c1, out_channels=chan_traj_c1_d0, ksize=4, stride=2)
        traj_d0 = L.Linear(in_size=chan_traj_c1_d0, out_size=units_traj_d0_d1)
        loss_d0 = L.Linear(in_size=traj_dim_in + units_traj_d0_d1, out_size=units_traj_d1_d2)
        loss_d1 = L.Linear(in_size=units_traj_d1_d2, out_size=1)

        Loss.__init__(self,
                      # trajectory processing
                      traj_c0=traj_c0, traj_c1=traj_c1, traj_d0=traj_d0,
                      # loss processing
                      loss_d0=loss_d0, loss_d1=loss_d1)

    def process_trajectory(self, l):
        """This is the time-dependent convolution operation, applied to a trajectory (in order).
        """
        shp = l.shape[0]
        # First dim is batchsize=1, then either 1 channel for 2d conv or n_feat channels
        # for 1d conv.
        l = F.expand_dims(l, axis=0)
        l = F.transpose(l, (0, 2, 1))
        l = self.traj_c0(l)
        l = F.leaky_relu(l)
        l = self.traj_c1(l)
        l = F.leaky_relu(l)
        l = F.sum(l, axis=(0, 2)) / l.shape[0] / l.shape[2]
        l = F.expand_dims(l, axis=0)
        l = self.traj_d0(l)
        l = F.tile(l, (shp, 1))
        return l

    def loss(self, l):
        """Loss function, can be calculated over a minibatch of data (out of order).
        """
        # # Explicit addition of policy output.
        l = self.loss_d0(l)
        l = F.leaky_relu(l)
        # Transforms into scalar
        l = self.loss_d1(l)
        # Need to put this otherwise error with BLAS ...
        l = l[:]
        # Average accross minibatch
        l = F.sum(l) / l.size
        return l
