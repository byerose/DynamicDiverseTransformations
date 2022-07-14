import sys
from time import process_time_ns
sys.path.append('/home/ait/ws/dynamic-preprocessor')
#sys.path.append(r'D:\\WS\dynamic-preprocessor')

import torch
import torch.nn as nn
from torch.autograd import Function
import random

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

from utils.trans_pool_cifar10 import *

trans = Transforms()
trans_dict = {}
trans_dict[trans.morphology] = TRANSFORMATION.MORPHOLOGY
trans_dict[trans.shift] = TRANSFORMATION.SHIFT
trans_dict[trans.rotate] = TRANSFORMATION.ROTATE
trans_dict[trans.filter] = TRANSFORMATION.FILTERS
trans_dict[trans.affine] = TRANSFORMATION.AFFINE
trans_dict[trans.compress] = TRANSFORMATION.COMPRESSION
trans_dict[trans.denoise] = TRANSFORMATION.DENOISE
trans_dict[trans.adjust] = TRANSFORMATION.ADJUSTMENT
trans_dict[trans.geometric] = TRANSFORMATION.GEOMETRIC


class Rotate(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                # Defense can be run on CPU after transferring data (shown here), but can also stay on device
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.rotate],1)[0]
                result = trans.rotate(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                # Apply BPDA or other gradient estimation
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y

class Adjusment(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                # Defense can be run on CPU after transferring data (shown here), but can also stay on device
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.adjust],1)[0]
                result = trans.adjust(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                # Apply BPDA or other gradient estimation
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y

class Affine(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                # Defense can be run on CPU after transferring data (shown here), but can also stay on device
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.affine],1)[0]
                result = trans.affine(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                # Apply BPDA or other gradient estimation
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y

class Compression(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.compress],1)[0]
                result = trans.compress(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y

class Denoise(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.denoise],1)[0]
                result = trans.denoise(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y

class Filters(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.filter],1)[0]
                result = trans.filter(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y


class Geometric(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.geometric],1)[0]
                result = trans.geometric(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y

class Morphology(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.morphology],1)[0]
                result = trans.morphology(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y


class Shift(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        class Preprocess(Function):  # pylint: disable=W0223
            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                numpy_input = input.detach().cpu().numpy()
                param = random.sample(trans_dict[trans.shift],1)[0]
                result = trans.shift(numpy_input,param)
                return input.new(result)
            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                result = numpy_go
                return grad_output.new(result)
        self._preprocess = Preprocess
    def forward(self, x, y=None):
        x.detach()
        x_bpda = self._preprocess.apply(x)
        return x_bpda, y
