from torch import nn
import torch.nn.functional as F


# LeNet that suppose Grad-CAM
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

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

        flat_vec = res['p2'].flatten(start_dim=1)
        res['logit'] = self.classifier(flat_vec)

        return res

    def get_activations_gradient(self):
        return self.gradients
