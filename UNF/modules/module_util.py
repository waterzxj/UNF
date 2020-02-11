#coding:utf-8
import torch.nn.init as init
import torch.nn as nn
import torch

from base_type import InitType, FAN_MODE, ActivationType

def init_tensor(tensor, init_type=InitType.XAVIER_UNIFORM, low=0, high=1,
                mean=0, std=1, activation_type=ActivationType.NONE,
                fan_mode=FAN_MODE.FAN_IN, negative_slope=0):
    """
    各种标准的tensor参数初始化方法
    """
    if init_type == InitType.UNIFORM:
        return torch.nn.init.uniform_(tensor, a=low, b=high)
    elif init_type == InitType.NORMAL:
        return torch.nn.init.normal_(tensor, mean=mean, std=std)
    elif init_type == InitType.XAVIER_UNIFORM:
        return torch.nn.init.xavier_uniform_(
            tensor, gain=torch.nn.init.calculate_gain(activation_type))
    elif init_type == InitType.XAVIER_NORMAL:
        return torch.nn.init.xavier_normal_(
            tensor, gain=torch.nn.init.calculate_gain(activation_type))
    elif init_type == InitType.KAIMING_UNIFORM:
        return torch.nn.init.kaiming_uniform_(
            tensor, a=negative_slope, mode=fan_mode,
            nonlinearity=activation_type)
    elif init_type == InitType.KAIMING_NORMAL:
        return torch.nn.init.kaiming_normal_(
            tensor, a=negative_slope, mode=fan_mode,
            nonlinearity=activation_type)
    elif init_type == InitType.ORTHOGONAL:
        return torch.nn.init.orthogonal_(
            tensor, gain=torch.nn.init.calculate_gain(activation_type))
    else:
        raise TypeError(
            "Unsupported tensor init type: %s. Supported init type is: %s" % (
                init_type, InitType.str()))

def initial_parameter(net, initial_method=None):
    """A method used to initialize the weights of PyTorch models.
    :param net: a PyTorch model
    :param str initial_method: one of the following initializations.
            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform
    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    net.apply(weights_init)
