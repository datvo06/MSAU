import torch


def get_optimizer(model, kwargs={}):
    params = model.parameters()
    optimizer_name = kwargs.get("optimizer", "rmsprop")
    learning_rate = kwargs.get("learning_rate", 0.001)
    lr_decay_rate = kwargs.get("lr_decay_rate", 0.0)
    if optimizer_name == "momentum":

        momentum = kwargs.get("momentum", 0.9)
        optimizer = torch.optim.SGD(params, learning_rate, momentum,
                                    weight_decay=lr_decay_rate)
    elif optimizer_name is "rmsprop":
        optimizer = torch.optim.RMSprop(
            params, learning_rate, weight_decay=lr_decay_rate)
    else:
        # Use Adam
        if learning_rate is not None:
            optimizer = torch.optim.Adam(params, learning_rate,
                                         weight_decay=lr_decay_rate)
        else:
            optimizer = torch.optim.Adam(params)

    print("Optimizer: " + optimizer_name)
    if learning_rate is None:
        print("Learning Rate: ")
    else:
        print("Learning Rate: " + str(learning_rate))
    return optimizer
