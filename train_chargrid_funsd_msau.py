from __future__ import print_function, division, unicode_literals
import torch, pickle, time, matplotlib, os, json
import torch.nn as nn
from torch.autograd import Variable
from data_generator_funsd_bert import FUNSDCharGridDataLoaderBoxMaskBoxLabel
import utils.io_utils as io_utils
import numpy as np
import sklearn.metrics as metrics
import random
from model.model import MSAUWrapper as MSAU


device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')


def train(dataset, model_instance, args, same_feat=True,
          val_dataset=None,
          test_dataset=None,
          writer=None,
          mask_nodes=True,
          ):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_instance.parameters()), lr=0.0001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    model_instance = model_instance.to(device)
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model_instance.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model_instance.zero_grad()
            all_feats = data["mask"]
            all_labels = data["label"]

            V = Variable(data["mask"].float(), requires_grad=False)  # .cuda()
            label = Variable(data["label"].long())  # .cuda()

            _, ypred, ypred_aux = model_instance(V.to(device))
            predictions += ypred.cpu().detach().numpy().tolist()

            loss = model_instance.loss(ypred, ypred_aux, label.to(device))
            loss.backward()
            nn.utils.clip_grad_norm(model_instance.parameters(), args.clip)
            optimizer.step()
            if batch_idx % 10 == 0:
                print("Batch {} optimized. Loss: {}" .format(
                    batch_idx, loss.cpu().detach().numpy()))
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(
            dataset, model_instance, args, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model_instance,
                                  args, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model_instance, args, testing=True, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])
        if epoch %10 == 0:
            filename = io_utils.create_filename(args.ckptdir, args, False, epoch)
            torch.save(model_instance.state_dict(), filename)
    matplotlib.style.use("seaborn")
    matplotlib.style.use("default")

    print("Shapes of \'all_feats\', \'all_labels\':",
          all_feats.shape,
          all_labels.shape, sep="\n")

    cg_data = {
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model_instance, optimizer, args, num_epochs=-1,
                             cg_dict=cg_data)
    return model_instance, val_accs


def evaluate(dataset, model, args,
             name="Validation", testing=False,
             max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        h0 = Variable(data["mask"].float())  # .cuda()
        instance_label = np.squeeze(data["label"].long().numpy())

        # TODO: fix the evaluate.
        _, ypred, _ = model.forward(h0.to(device))
        ypred = ypred.squeeze()
        _, indices = torch.max(ypred, 0) # H x W
        indices = indices.cpu().data.numpy()
        indices = indices[instance_label != 0]
        instance_label = instance_label[instance_label != 0 ]
        if testing:
            indices[indices == 0] = dataset.labels['other']
        labels.append(instance_label)
        preds.append(indices)

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels).squeeze()
    print("Label: ", labels.shape)
    preds = np.hstack(preds).squeeze()
    print("Predict: ", preds.shape)

    result = {
        # "prec": metrics.precision_score(labels, preds, average="macro"),
        # "recall": metrics.recall_score(labels, preds, average="macro"),
        "prec": metrics.precision_score(labels, preds, average="micro"),
        "recall": metrics.recall_score(labels, preds, average="micro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    if testing:
        print(metrics.classification_report(labels, preds, target_names=list(dataset.labels.keys())))
    print(name, " accuracy:", result["acc"])
    return result


class dummyArgs(object):
    def __init__(self):
        self.batch_size = 1
        self.bmname = None
        self.hidden_dim = 500
        self.dataset = "invoice"
        pass


if __name__ == '__main__':
    random.seed(777)

    data_loader = FUNSDCharGridDataLoaderBoxMaskBoxLabel(
        "./funsd_preprocess.pkl")
    data_loader_test = FUNSDCharGridDataLoaderBoxMaskBoxLabel(
        "./funsd_preprocess_test.pkl", data_loader.labels)
    # set up the arguments
    args = dummyArgs()
    args.output_dim = len(data_loader.labels.keys()) + 1
    args.clip = True
    args.ckptdir = "ckpt"
    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)
    args.model_kwargs_path = None
    args.method = "GCN"
    args.name = "dummy name"
    args.num_epochs = 300
    args.train_ratio = 0.8
    args.test_ratio = 0.0
    args.gpu = torch.cuda.is_available()

    # data_loader = PerGraphNodePredDataLoader("../Invoice_k_fold/save_features/all/input_features.pickle")

    i = 0
    feature_dim = data_loader[i]['mask'].shape[1]
    n_labels = args.output_dim

    model_kwargs_path = args.model_kwargs_path

    if model_kwargs_path is None:
        img_channels = feature_dim
        n_class = args.output_dim
        use_auxiliary_loss = False

        ''' model hyper-parameters '''
        model_kwargs = dict(model="msau", final_act="softmax",
                            featRoot=8, scale_space_num=4,
                            res_depth=2, n_class=n_class, img_channels=feature_dim,
                            use_auxiliary_loss=use_auxiliary_loss)
        json.dump(model_kwargs, open('model_kwargs.json', 'w'))
    else:
        model_kwargs = json.load(open(model_kwargs_path, 'r'))
        img_channels = model_kwargs['img_channels']
        n_class = model_kwargs['n_class']
        use_auxiliary_loss = model_kwargs['use_auxiliary_loss']

    model = MSAU(feature_dim, n_labels, model_kwargs=model_kwargs)
    model.to(device)
    instances = data_loader
    test_instances = data_loader_test
    indices = list(range(len(instances)))
    random.shuffle(indices)
    if test_instances is None:
        train_idx = int(len(instances) * args.train_ratio)
        test_idx = int(len(instances) * (1 - args.test_ratio))
        # train_graphs = graphs[indices[:train_idx]]
        train_instances = [instances[i] for i in indices[:train_idx]]
        # val_graphs = graphs[indices[train_idx:test_idx]]
        val_instances = [instances[i] for i in indices[train_idx:test_idx]]
        # test_graphs = graphs[indices[test_idx:]]
        test_instances = [instances[i] for i in indices[test_idx:]]
    else:
        train_idx = int(len(instances) * args.train_ratio)
        train_instances= [instances[i] for i in indices[:train_idx]]
        val_instances = [instances[i] for i in indices[train_idx:]]
    print(
        "Num training instances: ",
        len(train_instances),
        "; Num validation instances: ",
        len(val_instances),
        "; Num testing instances: ",
        len(instances),
    )

    train(train_instances,
          model_instance=model,
          args=args,
          same_feat=True,
          val_dataset=val_instances,
          test_dataset=test_instances,
          writer=None,
          mask_nodes=True,
          )

    print("Finished\n\n")
    # pass
