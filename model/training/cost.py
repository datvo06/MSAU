import torch
import torch.nn.functional as F
import numpy as np


class UNetLoss(torch.nn.Module):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """

    def __init__(self, kwargs):
        super(UNetLoss, self).__init__()
        self.cost_name = kwargs.get("cost_name", "cross_entropy")
        act_name = kwargs.get("act_name", "softmax")
        if act_name == "softmax":
            self.act = torch.nn.Softmax
        if act_name == "sigmoid":
            self.act = torch.nn.Sigmoid
        if act_name == "identity":
            self.act = torch.nn.Sequential()
        self.class_weights = kwargs.get("class_weights", None)

        if self.cost_name == 'cross_entropy':
            if self.class_weights is not None:
                self.class_weights_torch = torch.from_numpy(
                    np.array(self.class_weights, dtype=np.float32))
                self.criterion = torch.nn.CrossEntropyLoss(
                    self.class_weights_torch)
            else:
                self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits, tgt, kwargs):
        aux_logits = kwargs.get("aux_logits", None)
        aux_tgt = kwargs.get("aux_tgt", None)
        if self.cost_name != "cross_entropy":
            prediction = self.act(logits)
            return prediction
        tgt = torch.argmax(tgt, dim=1)
        pred = torch.argmax(logits, dim=1)
        with torch.no_grad():
            npy_tgt = tgt.cpu().data.numpy()
            npy_pred = pred.cpu().data.numpy()
            non_zero_mask = npy_tgt != 0
            acc = np.sum(
                npy_tgt[non_zero_mask] == npy_pred[non_zero_mask]
            ).astype('float') / (
                    np.sum(non_zero_mask)
            )
        if aux_tgt is not None:
            aux_tgt = torch.argmax(aux_tgt, dim=1)

        loss_map = self.criterion(logits, tgt)
        if aux_logits is not None:
            loss_map_aux = self.criterion(aux_logits, aux_tgt)
            final_loss = loss_map
            aux_loss = loss_map_aux
            # print("Aux loss: ", aux_loss, "\nFinal Loss: ", final_loss)
            loss = 0.5 * final_loss + 0.5 * aux_loss
        else:
            loss = loss_map
            final_loss = None
        return acc, loss, final_loss


def get_weighted_mean(mses, sums, globSum):
    loss = 0.0
    for aCH in range(0, len(sums)):
        aLoss = (1.0-sums[aCH]/globSum) * mses[aCH]
        loss += aLoss
    return loss
