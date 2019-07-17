from torch import nn
import torch.nn.functional as F


# Simple LeAE that suppose Grad-CAM
class LeAE(nn.Module):
    def __init__(self):
        super(LeAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 6, 5, stride=1)
        self.conv5 = nn.ConvTranspose2d(6, 1, 4, stride=2, padding=1)

        self.gradients = None

    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        res = dict()
        res['c1'] = F.relu(self.conv1(x))
        res['p1'] = self.pool1(res['c1'])
        res['c2'] = F.relu(self.conv2(res['p1']))
        res['c2'].register_hook(self.activation_hook)
        res['p2'] = self.pool2(res['c2'])

        res['c3'] = F.relu(self.conv3(res['p2']))
        res['c4'] = F.relu(self.conv4(res['c3']))
        res['rec'] = F.tanh(self.conv5(res['c4']))
        return res

    def get_activations_gradient(self):
        return self.gradients


class LeClassifier(nn.Module):
    def __init__(self):
        super(LeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.classifier(x)


class LeDiscriminator(nn.Module):
    def __init__(self):
        super(LeDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, stride=1)
        )

    def forward(self, x):
        res = self.net(x)
        return res.view(-1, 1)