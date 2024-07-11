"""Module"""
import torch
from torch import nn, optim
from torch.nn import init


class MLP(nn.Module):
    """MLP with two-way line search"""

    def __init__(self, n_inputs, n_classes, device, **params):
        """define model elements"""
        super().__init__()
        self.__dict__.update(params)
        self.device = device
        self.out_act = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        # MLP
        self.model = self.build_layers(n_inputs, n_classes)
        # to gpu is available
        self.to(device)

    def build_layers(self, n_inputs, n_classes):
        """Build architecture"""
        layer_sizes = [
            n_inputs,
            *self.layers,
        ]
        # build layers
        layers_mlp = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            linear_layer_mlp = nn.Linear(in_, out_)
            # weight initialization
            if self.activation == "relu":
                init.kaiming_normal_(linear_layer_mlp.weight, nonlinearity="relu")
            else:
                init.xavier_normal_(linear_layer_mlp.weight)
            # adding layers to lists
            layers_mlp.append(linear_layer_mlp)
            # adding activations
            layers_mlp.append(self.select_activation())
        # Output layers
        layers_mlp.append(nn.Linear(layer_sizes[-1], n_classes))
        init.xavier_normal_(layers_mlp[-1].weight)
        return nn.Sequential(*layers_mlp)

    def select_activation(self):
        """Produces the activation function based on the choice"""
        if self.activation == "tanh":
            return nn.Tanh()
        if self.activation in ("sigmoid", "logistic"):
            return nn.Sigmoid()
        return nn.ReLU()

    def forward(self, X):
        """Forward propagate input"""
        X = self.model(X)
        return X

    def select_optimizer(self):
        """Select training algorithm. Some of these are optional
        so we can't use dictionaries"""
        optimizer = optim.AdamW(
                self.parameters(),
                betas=(self.beta_1, self.beta_2),
                lr=self.lr,
                weight_decay=self.weight_decay,
        )
        return optimizer

    # train the model
    def train_evaluate(self, train_dl):
        """Test then train"""
        # define the optimization algorithm
        optimizer = self.select_optimizer()
        preds = torch.empty(0).to(self.device)
        true_labels = torch.empty(0).to(self.device)
        losses = torch.empty(0).to(self.device)
        # enumerate instances
        for time_step, (inputs, targets) in enumerate(train_dl):
            # clear the gradients, we clear them for each instance - strict online
            optimizer.zero_grad()
            # compute the model output
            yhat = self(inputs)
            true_labels = torch.cat((true_labels, targets))
            # calculate loss
            loss = self.criterion(yhat, targets)
            # calculate gradient
            loss.backward()
            # update model weights
            optimizer.step()
            preds = torch.cat(
                (preds, torch.argmax(self.out_act(yhat.detach()), axis=1))
            )
            losses = torch.cat(
                (
                    losses,
                    loss.detach().reshape(1),
                )
            )
        return preds, true_labels, losses

    def predict(self, instance):
        """make a class prediction for one instance of data"""
        yhat = self(instance)
        return yhat
