class InitType():
    """
    各种矩阵初始化的方法
    """
    UNIFORM = 'uniform'
    NORMAL = "normal"
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    ORTHOGONAL = 'orthogonal'

    def __str__(self):
        return ",".join(
            [self.UNIFORM, self.NORMAL, self.XAVIER_UNIFORM, self.XAVIER_NORMAL,
             self.KAIMING_UNIFORM, self.KAIMING_NORMAL, self.ORTHOGONAL])


class FAN_MODE():
    FAN_IN = 'FAN_IN'
    FAN_OUT = "FAN_OUT"

    def __str__(self):
        return ",".join([self.FAN_IN, self.FAN_OUT])


class ActivationType():
    SIGMOID = 'sigmoid'
    TANH = "tanh"
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    NONE = 'linear'

    def __str__(self):
        return ",".join(
            [self.SIGMOID, self.TANH, self.RELU, self.LEAKY_RELU, self.NONE])
