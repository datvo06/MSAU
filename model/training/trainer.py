from __future__ import print_function, division

import torch
import os
import time

from .cost import UNetLoss
from .optimizer import get_optimizer
device = torch.device("cuda:0")


class Trainer(object):
    """
    Trains a MSAU-net instance

    :param net: the arunet instance to train
    :param opt_kwargs: (optional) kwargs passed to the optimizer
    :param cost_kwargs: (optional) kwargs passed to the cost function

    """

    def __init__(self, net, opt_kwargs={}, cost_kwargs={}):
        self.net = net
        self.opt_kwargs = opt_kwargs

        self.use_auxiliary_loss = cost_kwargs.get("use_auxiliary_loss", True)
        if self.use_auxiliary_loss:
            self.cost_kwargs = {"aux_logits": None, "aux_tgt": None}
        else:
            self.cost_kwargs = cost_kwargs

        self.cost_type = cost_kwargs.get("cost_name", "cross_entropy")
        self.criterion = UNetLoss(self.cost_kwargs)

    def _initialize(self, output_path):
        self.optimizer = get_optimizer(
            self.net, self.opt_kwargs)

        if output_path is not None:
            output_path = os.path.abspath(output_path)
            if not os.path.exists(output_path):
                print("Allocating '{:}'".format(output_path))
                os.makedirs(output_path)

    def adjust_lr(self, epoch):
        lr = 0.001 * (0.95 ** (epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def load_weights(self, weights_dict, output_path):
        save_path = os.path.join(output_path, "model")
        self.net.load_weights(weights_dict)
        save_pathAct = save_path + "02"
        self.net.save(save_pathAct)

    def train(self, data_provider, output_path, restore_path=None,
              batch_steps_per_epoch=1024, epochs=250,
              gpu_device="0", max_spat_dim=5000000):
        """
        Launches the training process
        :param data_provider:
        :param output_path:
        :param restore_path:
        :param batch_size:
        :param batch_steps_per_epoch:
        :param epochs:
        :param keep_prob:
        :param gpu_device:
        :param max_spat_dim:
        :return:
        """
        print("Epochs: " + str(epochs))
        print("Batch Size Train: " + str(data_provider.batchsize_tr))
        print("Batchsteps per Epoch: " + str(batch_steps_per_epoch))
        if output_path is not None:
            save_path = os.path.join(output_path, "model")
        if epochs == 0:
            return save_path

        self._initialize(output_path)

        val_size = data_provider.size_val

        if restore_path is not None:
            print("Loading Checkpoint.")
            self.net.load_weights(restore_path)
        print("Starting from scratch.")

        print("Start optimization")

        bestLoss = 100000.0
        shown_samples = 0
        for epoch in range(epochs):
            torch.cuda.empty_cache()
            lr = self.adjust_lr(epoch)
            total_loss = 0
            total_loss_final = 0
            time_step_train = time.time()
            avg_acc = []
            self.net.train()
            for step in range((epoch * batch_steps_per_epoch),
                              ((epoch + 1) * batch_steps_per_epoch)):
                batch_x, batch_tgt, batch_tgt_aux =\
                    data_provider.next_data('train')
                skipped = 0
                if batch_x is None:
                    print("No Training Data available. Skip \
                          Training Path.")
                    break
                batch_x = batch_x.float().cuda()
                batch_tgt = batch_tgt.long().cuda()
                batch_tgt_aux = batch_tgt_aux.long().cuda()
                while batch_x.size()[2] * batch_x.size()[3] > max_spat_dim:
                    batch_x, batch_tgt = data_provider.next_data('train')
                    skipped = skipped + 1
                    if skipped > 100:
                        print("Spatial Dimension of Training Data to \
                              high. Aborting.")
                        return save_path
                # Run training
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    predict_result, logits, logits_aux = self.net(batch_x)
                    # logits = logits.transpose(1, -1).transpose(1, 2)
                    # logits_aux = logits_aux.transpose(
                    #    1, -1).transpose(1, 2)
                    if self.use_auxiliary_loss:
                        self.cost_kwargs['aux_logits'] = logits_aux
                    else:
                        self.cost_kwargs['aux_logits'] = None
                    self.cost_kwargs['aux_tgt'] = batch_tgt_aux
                    acc, loss, final_loss = self.criterion(logits, batch_tgt,
                                                self.cost_kwargs)
                    avg_acc.append(acc)
                    loss.backward()
                    self.optimizer.step()
                    if final_loss is not None:
                        total_loss_final += final_loss
                    shown_samples = shown_samples + batch_x.size()[0]
                    if self.cost_type is "cross_entropy_sum":
                        shape = batch_x.shape
                        loss /= shape[3] * shape[2] * shape[0]
                    total_loss += loss.item()

            total_loss = total_loss / batch_steps_per_epoch
            total_loss_final = total_loss_final / batch_steps_per_epoch
            time_used = time.time() - time_step_train
            self.output_epoch_stats_train(epoch + 1, sum(avg_acc)/len(avg_acc),  total_loss,
                                          total_loss_final, shown_samples,
                                          lr, time_used)
            total_loss = 0
            total_loss_final = 0
            time_step_val = time.time()
            avg_acc = []
            self.net.eval()
            for step in range(0, val_size):
                batch_x, batch_tgt, batch_tgt_aux =\
                    data_provider.next_data('val')
                batch_x = batch_x.float().cuda()
                batch_tgt = batch_tgt.long().cuda()
                batch_tgt_aux = batch_tgt_aux.long().cuda()

                if batch_x is None:
                    print("No Validation Data available. \
                          Skip Validation Path.")
                    break
                # Run validation
                predict_result, logits, logits_aux = self.net(batch_x)
                # logits = logits.transpose(1, -1).transpose(1, 2)
                # logits_aux = logits_aux.transpose(
                #        1, -1).transpose(1, 2)
                if self.use_auxiliary_loss:
                    self.cost_kwargs['aux_logits'] = logits_aux
                else:
                    self.cost_kwargs['aux_logits'] = None
                self.cost_kwargs['aux_tgt'] = batch_tgt_aux
                acc, loss, final_loss = self.criterion(logits, batch_tgt,
                                            self.cost_kwargs)
                avg_acc.append(acc)
                if final_loss is not None:
                    total_loss_final += final_loss.item()
                if self.cost_type is "cross_entropy_sum":
                    shape = batch_x.shape
                    loss /= shape[3] * shape[2] * shape[0]
                total_loss += loss.item()
            if val_size != 0:
                total_loss = total_loss / val_size
                total_loss_final = total_loss_final / val_size
                time_used = time.time() - time_step_val
                self.output_epoch_stats_val(
                    epoch + 1, sum(avg_acc)/len(avg_acc),
                    total_loss, total_loss_final, time_used)
                data_provider.restart_val_runner()

            if output_path is not None:
                if total_loss < bestLoss or (epoch + 1) % 8 == 0:
                    if total_loss < bestLoss:
                        bestLoss = total_loss
                    save_pathAct = save_path + str(epoch + 1)
                    print("Saving checkpoint")
                    self.net.save(save_pathAct)

        data_provider.stop_all()
        print("Optimization Finished!")
        print("Best Val Loss: " + str(bestLoss))
        return save_path

    def output_epoch_stats_train(self, epoch, acc, total_loss,
                                 total_loss_final, shown_sample, lr,
                                 time_used):
        print(
            "TRAIN: Epoch {:}, Average Acc: {:.6f}, Average loss: {:.6f}  final: {:.6f}, \
            training samples shown: {:}, \
            learning rate: {:.6f}, time used: {:.2f}".format(
                epoch, acc, total_loss, total_loss_final,
                shown_sample, lr, time_used))

    def output_epoch_stats_val(self, epoch, acc, total_loss,
                               total_loss_final, time_used):
        print(
                "VAL: Epoch {:}, Average Acc: {:.6f}, Average loss: {:.6f} \
            final: {:.6f}, time used: {:.2f}".format(epoch, acc, total_loss,
                                                     total_loss_final,
                                                     time_used))
