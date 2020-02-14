#coding:utf-8
import os
import shutil
import re
import logging
import json

import torch
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class Checkpointer(object):
    def __init__(self,
                 serialization_dir,
                 num_serialized_models_to_keep):
        """
        模型状态和训练状态的追踪，负责保存和训练恢复
        """
        self.serialization_dir = serialization_dir
        self.num_serialized_models_to_keep = num_serialized_models_to_keep

    def save_checkpoint(self,
                        epoch,
                        model_state,
                        training_states,
                        is_best_so_far):
        if self.serialization_dir is not None:
            if epoch < self.num_serialized_models_to_keep or is_best_so_far:
                model_path = os.path.join(self.serialization_dir, "model_state_epoch_{}.th".format(epoch))
                torch.save(model_state, model_path)
                training_path = os.path.join(self.serialization_dir,
                                         "training_state_epoch_{}.th".format(epoch))
                torch.save({**training_states, "epoch": epoch}, training_path)

            if is_best_so_far:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self.serialization_dir)
                shutil.copyfile(model_path, os.path.join(self.serialization_dir, "best.th"))

    def find_latest_checkpoint(self):
        """
        返回模型存储路径的上一次保存的模型路径和训练状态的路径
        """
        have_checkpoint = (self.serialization_dir is not None and
                           any("model_state_epoch_" in x for x in os.listdir(self.serialization_dir)))

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self.serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            re.search(r"model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
            for x in model_checkpoints
        ]
        int_epochs = []
        for epoch in found_epochs:
            pieces = epoch.split('.')
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), '0'])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == '0':
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(self.serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self.serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        return (model_path, training_state_path)

    def restore_checkpoint(self):
        """
        用于模型的恢复训练，如果只是想加载模型的checkpoint，建议还是用torch原始的方式model.load_state_dict()
        """
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            return {}, {}

        model_path, training_state_path = latest_checkpoint

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(model_path, "cpu")
        training_state = torch.load(training_state_path, "cpu")
        return model_state, training_state

    def best_model_state(self):
        if self.serialization_dir:
            logger.info("loading best weights")
            best_model_state_path = os.path.join(self.serialization_dir, 'best.th')
            return torch.load(best_model_state_path)
        else:
            logger.info("cannot load best weights without `serialization_dir`, "
                        "so you're just getting the last weights")
            return {}


class MetricTracker(object):
    def __init__(self,
                 patience,
                 metric_name,
                 should_decrease=None):
        """
        模型训练metric管理，earlystopping管理

        :prams patience int earlystopping设置，epoch相关
        :params metric_name str 需要评估的模型指标
        :params should_decrease bool 指标是越大越好，还是越小越好
        """
        self._best_so_far = None
        self._patience = patience
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self.best_epoch_metrics = {}
        self._epoch_number = 0
        self.best_epoch = None

        if metric_name is not None and should_decrease is not None:
            raise Exception("Metric and should_decrease should not 1same none")

        if should_decrease != None:
            self._should_decrease = should_decrease

        elif metric_name is not None:
            if metric_name[0] == "-":
                self._should_decrease = True
            elif metric_name[0] == "+":
                self._should_decrease = False
            else:
                raise Exception("metric_name must start with + or -")

    def clear(self):
        """
        状态清除
        """
        self._best_so_far = None
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self._epoch_number = 0
        self.best_epoch = None

    def state_dict(self):
        """
        训练状态的保存
        """
        return {
            "best_so_far": self._best_so_far,
            "patience": self._patience,
            "epochs_with_no_improvement": self._epochs_with_no_improvement,
            "is_best_so_far": self._is_best_so_far,
            "should_decrease": self._should_decrease,
            "best_epoch_metrics": self.best_epoch_metrics,
            "epoch_number": self._epoch_number,
            "best_epoch": self.best_epoch
        }

    def load_state_dict(self, state_dict):
        """
        训练状态的加载
        """
        self._best_so_far = state_dict["best_so_far"]
        self._patience = state_dict["patience"]
        self._epochs_with_no_improvement = state_dict["epochs_with_no_improvement"]
        self._is_best_so_far = state_dict["is_best_so_far"]
        self._should_decrease = state_dict["should_decrease"]
        self.best_epoch_metrics = state_dict["best_epoch_metrics"]
        self._epoch_number = state_dict["epoch_number"]
        self.best_epoch = state_dict["best_epoch"]

    def add_metric(self, metric):
        """
        加入metric指标，并更新相关状态
        """
        new_best = ((self._best_so_far is None) or
                    (self._should_decrease and metric < self._best_so_far) or
                    (not self._should_decrease and metric > self._best_so_far))

        if new_best:
            self.best_epoch = self._epoch_number
            self._is_best_so_far = True
            self._best_so_far = metric
            self._epochs_with_no_improvement = 0
        else:
            self._is_best_so_far = False
            self._epochs_with_no_improvement += 1
        self._epoch_number += 1

    def add_metrics(self, metrics):
        """
        添加多个metric
        """
        for metric in metrics:
            self.add_metric(metric)

    def is_best_so_far(self):
        """
        当前epoch是否是metric最好的
        """
        return self._is_best_so_far

    def should_stop_early(self):
        """
        是否需要earlystoping
        """
        if self._patience is None:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience


class TensorBoardWriter(object):
    def __init__(self,
                 get_batch_num_total,
                 serialization_dir=None,
                 summary_interval=100,
                 histogram_interval=None,
                 should_log_parameter_statistics=True,
                 should_log_learning_rate=False):
        """
        封装模型训练过程中相应的打点的输出（tensorboard|terminal)

        :params get_batch_num_total callable 返回模型训练目前的batch数,用于判断是否需要打点
        :params serialization_dir str 日志存放的目录
        :params summary_interval int 打点间隔的batch
        :params histogram_interval int 非None表示打柱状图的间隔batch
        :params should_log_parameter_statistics bool 是否打点关于参数的统计量的点
        :params should_log_learning_rate bool 是否打点关于学习率的统计量的点
        """  
        if serialization_dir is not None:
            self._train_log = SummaryWriter(os.path.join(serialization_dir, "log", "train"))
            self._validation_log = SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
        else:
            self._train_log = self._validation_log = None

        self._summary_interval = summary_interval
        self._histogram_interval = histogram_interval
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self._should_log_learning_rate = should_log_learning_rate
        self._get_batch_num_total = get_batch_num_total

    def _item(self, value):
        if hasattr(value, 'item'):
            val = value.item()
        else:
            val = value
        return val

    def should_log_this_batch(self):
        return self._get_batch_num_total() % self._summary_interval == 0

    def should_log_histograms_this_batch(self):
        return self._histogram_interval is not None and \
                self._get_batch_num_total() % self._histogram_interval == 0

    def add_train_scalar(self, name, value):
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), self._get_batch_num_total())
    
    def add_train_histogram(self, name, values):
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, self._get_batch_num_total())

    def add_validation_scalar(self, name, value):
        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), self._get_batch_num_total())

    def log_parameter_and_gradient_statistics(self,
                                              model,
                                              batch_grad_norm=None):
        """
        把模型的参数和梯度的统计量（均值和方差）打点到tensorboard
        """
        if self._should_log_parameter_statistics:
            # Log parameter values to Tensorboard
            for name, param in model.named_parameters():
                self.add_train_scalar("parameter_mean/" + name, param.data.mean())
                self.add_train_scalar("parameter_std/" + name, param.data.std())
                if param.grad is not None:
                    if param.grad.is_sparse:
                        grad_data = param.grad.data._values()
                    else:
                        grad_data = param.grad.data

                    # skip empty gradients
                    if torch.prod(torch.tensor(grad_data.shape)).item() > 0:  # pylint: disable=not-callable
                        self.add_train_scalar("gradient_mean/" + name, grad_data.mean())
                        self.add_train_scalar("gradient_std/" + name, grad_data.std())
                    else:
                        # no gradient for a parameter with sparse gradients
                        logger.info("No gradient for %s, skipping tensorboard logging.", name)
            # norm of gradients
            if batch_grad_norm is not None:
                self.add_train_scalar("gradient_norm", batch_grad_norm)

    def log_learning_rates(self,
                           model,
                           optimizer):
        """
        把当前模型的学习率打点到tensorboard
        """
        if self._should_log_learning_rate:
            # optimizer stores lr info keyed by parameter tensor
            # we want to log with parameter name
            names = {param: name for name, param in model.named_parameters()}
            for group in optimizer.param_groups:
                if 'lr' not in group:
                    continue
                rate = group['lr']
                for param in group['params']:
                    # check whether params has requires grad or not
                    effective_rate = rate * float(param.requires_grad)
                    self.add_train_scalar("learning_rate/" + names[param], effective_rate)

    def log_histograms(self, model, histogram_parameters):
        """
        模型柱状图的参数打点到tensorboardx

        :params histogram_parameters set|list[str] 打点的名字
        """
        for name, param in model.named_parameters():
            if name in histogram_parameters:
                self.add_train_histogram("parameter_histogram/" + name, param)

    def log_metrics(self,
                    train_metrics,
                    val_metrics=None,
                    log_to_console=False):
        """
        把训练和验证（如果提供了）的metric信息打点到tensorboardx
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        # For logging to the console
        if log_to_console:
            dual_message_template = "%s |  %8.3f  |  %8.3f"
            no_val_message_template = "%s |  %8.3f  |  %8s"
            no_train_message_template = "%s |  %8s  |  %8.3f"
            header_template = "%s |  %-10s"
            name_length = max([len(x) for x in metric_names])
            logger.info(header_template, "Training".rjust(name_length + 13), "Validation")

        train_confusion_matrix_metric, val_confusion_matrix_metric = None, None
        for name in metric_names:
            # Log to tensorboard
            if 'confusion_matrix' not in name:
                train_metric = train_metrics.get(name)
                if train_metric is not None:
                    self.add_train_scalar(name, train_metric)
                val_metric = val_metrics.get(name)
                if val_metric is not None:
                    self.add_validation_scalar(name, val_metric)
            else:
                train_confusion_matrix_metric = train_metrics.get(name)
                val_confusion_matrix_metric = val_metrics.get(name)
                continue

            # And maybe log to console
            if log_to_console and val_metric is not None and train_metric is not None:
                logger.info(dual_message_template, name.ljust(name_length), train_metric, val_metric)
            elif log_to_console and val_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", val_metric)
            elif log_to_console and train_metric is not None:
                logger.info(no_val_message_template, name.ljust(name_length), train_metric, "N/A")

        if log_to_console and train_confusion_matrix_metric is not None:
            logger.info("Training".rjust(name_length + 8))
            for index, items in enumerate(train_confusion_matrix_metric):
                logger.info("class%s |  %s", str(index).ljust(name_length),
                            '  |  '.join([str(x).center(8, ' ') for x in items]))
        if log_to_console and val_confusion_matrix_metric is not None:
            logger.info("Validation".rjust(name_length + 8))
            for index, items in enumerate(val_confusion_matrix_metric):
                logger.info("class%s |  %s", str(index).ljust(name_length),
                            '  |  '.join([str(x).center(8, ' ') for x in items]))

    def enable_activation_logging(self, model):
        """
        模型激活值输出的统计信息打点到tensorboardx
        """
        if self._histogram_interval is not None:
            #采用给每个module注册hook的方式，但是要实现每隔一定batch才输出的功能，所以采用了闭包的实现方式
            for _, module in model.named_modules():
                if not getattr(module, 'should_log_activations', False):
                    # skip it
                    continue

                def hook(module_, inputs, outputs):
                    # pylint: disable=unused-argument,cell-var-from-loop
                    log_prefix = 'activation_histogram/{0}'.format(module_.__class__)
                    if self.should_log_histograms_this_batch():
                        self.log_activation_histogram(outputs, log_prefix)
                module.register_forward_hook(hook)

    def log_activation_histogram(self, outputs, log_prefix):
        if isinstance(outputs, torch.Tensor):
            log_name = log_prefix
            self.add_train_histogram(log_name, outputs)
        elif isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                log_name = "{0}_{1}".format(log_prefix, i)
                self.add_train_histogram(log_name, output)
        elif isinstance(outputs, dict):
            for k, tensor in outputs.items():
                log_name = "{0}_{1}".format(log_prefix, k)
                self.add_train_histogram(log_name, tensor)
        else:
            # skip it
            pass


def rescale_gradients(model, grad_norm):
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters()
                              if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None


def sparse_clip_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad.is_sparse:
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


def enable_gradient_clipping(model, grad_clipping):
    if grad_clipping is not None:
        for parameter in model.parameters():
            if parameter.requires_grad:
                #tensor的register_hook在tensor的梯度计算完之后hook(grad)
                #ref:https://pytorch.org/docs/stable/autograd.html#torch.Tensor.register_hook
                parameter.register_hook(lambda grad: clamp_tensor(grad,
                                                                  minimum=-grad_clipping,
                                                                  maximum=grad_clipping))

def clamp_tensor(tensor, minimum, maximum):
    """
    Supports sparse and dense tensors.
    Returns a tensor with values clamped between the provided minimum and maximum,
    without modifying the original tensor.
    """
    if tensor.is_sparse:
        coalesced_tensor = tensor.coalesce()
        # pylint: disable=protected-access
        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)

def dump_metrics(file_path, metrics, log=False):
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)


def generate_mask(data_seq_length, seq_size, batch_size):
    """
    根据batch中每个sequence的实际长度和batch中最长sequence生成batch mask
    """
    mask = torch.zeros((batch_size, seq_size)).byte()
    for idx, length in enumerate(data_seq_length):
        mask[idx, :length] = torch.Tensor([1]*length)

    return mask

def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

    
