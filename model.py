"""
Main model (FedX) class representing backbone network and projection heads

"""

import torch.nn as nn

from resnetcifar import ResNet18_cifar10, ResNet50_cifar10, ResNet18_MNIST


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        h = y.view(y.shape[0], -1)
        proj = self.fc1(h)
        proj = self.relu3(proj)
        proj = self.fc2(proj)
        pred = self.relu4(proj)
        pred = self.fc3(pred)
        pred = self.relu5(pred)
        return h, proj, pred


class ModelFedX(nn.Module):
    def __init__(self, base_model, out_dim, net_configs=None, loss=None):
        super(ModelFedX, self).__init__()
        
        if (
                base_model == "resnet50-cifar10"
                or base_model == "resnet50-cifar100"
                or base_model == "resnet50-smallkernel"
                or base_model == "resnet50"
        ):
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-fmnist":
            basemodel = ResNet18_MNIST()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features
        else:
            raise ("Invalid model type. Check the config file and pass one of: resnet18 or resnet50")

        self.projectionMLP = nn.Sequential(
            nn.Linear(self.num_ftrs, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

        self.predictionMLP = nn.Sequential(
            nn.Linear(out_dim, 10),
            # nn.ReLU(inplace=True),
            # nn.Linear(out_dim, out_dim),
        )
        self.simsiampredictionMLP = lambda x: x
        if loss == 'simsiam':
            self.simsiampredictionMLP = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.BatchNorm1d(out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 2, out_dim)
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"

    def forward(self, x):
        h = self.features(x)

        h.view(-1, self.num_ftrs)
        h = h.squeeze()

        proj = self.projectionMLP(h)
        pred = self.predictionMLP(proj)
        predsiam = self.simsiampredictionMLP(proj)
        return h, proj, pred, predsiam


def init_nets(net_configs, n_parties, args, device="cpu", loss=None):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        if args.model == 'lenet':
            net = LeNet()
        else:
            net = ModelFedX(args.model, args.out_dim, net_configs, loss)
        net = net.cuda()
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type
