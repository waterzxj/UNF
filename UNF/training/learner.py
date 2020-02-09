#coding:utf-8
import logging
import time

from .util import MetricTracker,TensorboardWriter,rescale_gradients
from .util import dump_metrics, Checkpointer,enable_gradient_clipping
from .metric import *
from .loss import *

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, model, train_iter, val_iter,
                    loss, optimizer, num_epochs,
                    loss=None, metric="F1Measure", label_index=1,
                    serialization_dir=None,
                    patience=None, validation_metric="-loss",
                    device=-1, grad_norm=None, grad_clipping=None,
                    learning_rate_scheduler=None, summary_interval=100,
                    histogram_interval=None, should_log_parameter_statistics=True,
                    should_log_learning_rate=False, log_batch_size_period=None,
                    num_serialized_models_to_keep=20, **kwargs):          
        """
        训练过程的封装

        :params model 训练的模型 继承自models.model.Model类型
        :params loss 损失函数
        :params optimizer 优化器
        :params epoch 迭代次数
        :params serialization_dir 模型保存路径
        :params patience int 非None则会应用earlystoping机制
        :params validation_metric 模型保存和earlystopping看的模型指标
        :params grad_norm float 反向传播的梯度的范数和会被rescale到不超过这个值
        :params grad_clip float 反向传播的梯度会被截断到最大这个值
        :params summary_interval int 向tensorboard传递metric的batch间隔
        :params histogram_interval int 非None，则会每隔histogram_interval个batch将训练的直方图
                传给tensorboard
        :params should_log_parameter_statistics int 非None，则会每隔should_log_parameter_statistics个batch
                将参数的统计信息传递给tensorboard
        :params should_log_learning_rate bool 是否将训练的learningrate传给tensorboard
        :params log_batch_size_period int 将batch_size信息传递给tensorboard的频率
        :params num_serialized_models_to_keep int 模型的前多少epoch保存checkpoint，默认为20
        """
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.serialization_dir = serialization_dir
        self.cuda_device = cuda_device
        self.learning_rate_scheduler = learning_rate_scheduler

        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping

        self.patience = patience
        self.validation_metric = validation_metric[1:]
        self.num_serialized_models_to_keep = num_serialized_models_to_keep

        self.summary_interval = summary_interval
        self.histogram_interval = histogram_interval
        self.should_log_parameter_statistics = should_log_parameter_statistics
        self.should_log_learning_rate = should_log_learning_rate
        self.log_batch_size_period = log_batch_size_period
        self.batch_num_total = 0

        self.checkpointer = Checkpointer(serialization_dir, num_serialized_models_to_keep)
        self.metric_tracker = MetricTracker(patience, validation_metric)
        self.tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self.batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate)

        if histogram_interval is not None:
            self.tensorboard.enable_activation_logging(self.model)

        self.metric = globals()[metric](label_index)
        self.loss_func = globals()[loss]()

    def rescale_gradients(self):
        return rescale_gradients(self.model, self.grad_norm)

    def learn(self):
        """
        模型学习过程的模拟
        """
        try:
            epoch_counter = self.restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise Exception("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")
        
        enable_gradient_clipping(self.model, self.grad_clipping)
        logger.info("Beginning training.")
        training_elapsed_time = time.time()
        train_epoch = 0

        metrcis = {}
        metrics['best_epoch'] = self.metric_tracker.best_epoch
        for key, value in self.metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epcoh in range(epoch_counter, self.epoch):
            train_metrics = self.train_epoch(epcoh)
            #机器情况的监控
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            if self.val_iter is not None:
                with torch.no_grad():
                    val_loss, num_batches = self.val_epoch()
                    val_metrics = self.get_metrics(val_loss, num_batches, reset=True)
                    this_epoch_val_metric = val_metrics[self.validation_metric]
                    self.metric_tracker.add_metric(this_epoch_val_metric)

                    if self.metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self.tensorboard.log_metrics(train_metrics, val_metrics=val_metrics, log_to_console=True)
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = train_epoch
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self.metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self.metric_tracker.best_epoch_metrics = val_metrics

            if self.serialization_dir:
                dump_metrics(os.path.join(self.serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            if self.learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self.learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            self.save_checkpoint(epoch)
            train_epoch += 1

            return metrcis

    def save_checkpoint(self, epoch):
        """
        保存模型训练的checkpoint
        """
        #checkpoint内部状态的追踪
        training_states = {
            "metric_traceker": self.metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self.batch_num_total
        }
        if self.learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = (
                self.learning_rate_scheduler.state_dict()
            )

        self.checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self.metric_tracker.is_best_so_far()
        )

    def restore_checkpoint(self):
        """
        恢复模型的训练，从最近一次保存的模型恢复checkpoint
        """
        model_state, training_state = self.checkpointer.restore_checkpoint()
        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self.learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self.learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])

        #move_optimizer_to_cuda(self.optimizer) ？？？
        if "metric_tracker" in training_state:
            self.metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self.metric_tracker.clear()
            self.metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self.metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1
        
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self.batch_num_total = batch_num_total

        return epoch_to_return


    def train_epoch(self, epoch):
        """
        一个epoch的训练，并返回相应的训练metric
        """
        logger.info("Epoch %d/%d", epoch, self.num_epochs - 1)
        train_loss = 0.0

        self.model.train()
        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0
        
        histogram_parameters = set(self.model.get_parameter_names())
        logger.info("Training")

        for batch_group in self.train_data:
            batches_this_epoch += 1
            self._batch_num_total += 1
            self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group)
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()
            train_loss += loss.item()
            batch_grad_norm = self.rescale_gradients()

            if self.tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self.tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7))
            else:
                self.optimizer.step()

            if self.tensorboard.should_log_this_batch():
                self.tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self.tensorboard.log_learning_rates(self.model, self.optimizer)

                metrics = self.get_metrics(train_loss, batches_this_epoch)
                self.tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self.tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self.tensorboard.should_log_histograms_this_batch():
                self.tensorboard.log_histograms(self.model, histogram_parameters)

        metrics = self.get_metrics(-, batches_this_epoch, reset=True)
        return metrcis
        
    def get_metrics(self, loss, batchs, reset=False):
        metrics = self.metric.get_metrics(reset)
        metrcis["loss"] = loss / (batchs + 1e-8)
        return metrcis

    def batch_loss(self, batch_group):
        """
        每个batch的数据得到loss
        """
        data, label = batch_group
        logits = self.model(data)["logits"]
        #计算loss
        loss = self.loss_func(logits, label)
        #更新metric
        self.metric(logits, label)
        return loss

    def val_epoch(self):
        logger.info("Validating")
        self.model.eval()
        batches_this_epoch = 0
        val_loss = 0.0

        for batch_group in self.val_iter:
            loss = self.batch_loss(batch_group)

            is loss is not None:
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

        return val_loss, batches_this_epoch
    
    @classmethod
    def from_options(cls, conf)
        return cls(**conf)
