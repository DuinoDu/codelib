import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
import os, glob

class FCN32s(nn.Module):

    def __init__(self, n_classes, phase, pretrain=False):
        super(FCN32s, self).__init__()
        self.n_classes = n_classes
        self.phase = phase

        self.base_net = self._base_net()
        self.predict_net = self._predict_net()

        self._init_weight()
        if pretrain:
            self._load_weight()


    def _base_net(self):
        sequential = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        return sequential 

    def _predict_net(self):
        classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),)
        return classifier 

    def forward(self, x):
        conv5 = self.base_net(x)
        score = self.predict_net(conv5)
        out = F.upsample(score, x.size()[2:], mode='bilinear')
        return out

    def _init_weight(self):
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.apply(weight_init)

    def _load_weight(self, weight_file=None, copy_fc8=False):

        def _fetch_weight():
            print('Fetching pretrained model...')
            vgg16 = models.vgg16(pretrained=True)
            model_file = os.path.join(os.environ['HOME'], '.torch/models', 'vgg16-*.pth')
            return glob.glob(model_file)[0]

        if weight_file == None:
            weight_file = _fetch_weight()
    
        _, ext = os.path.splitext(weight_file)
        if ext not in ('.pkl', '.pth'):
            raise ValueError, 'Sorry, only .pth and .pkl are supported, get {}'.format(weight_file)

        source_dict = torch.load(weight_file)
        keys = source_dict.keys()
        for key in keys:
            if 'features' in key:
                source_dict['base_net'+key[8:]] = source_dict.pop(key)
            elif 'classifier' in key:
                new_key = 'predict_net'+key[10:]
                new_size = self.state_dict()[new_key].size()
                if not copy_fc8 and 'classifier.6' in key: # skip fc8, "classifier.6"
                    source_dict.pop(key)
                    continue
                source_dict[new_key] = source_dict.pop(key).view(new_size)

        self.load_state_dict(source_dict, strict=False)
        print('Loading weight successfully!')


if __name__ == "__main__":
    from torch.autograd import Variable
    net = FCN32s(21, 'train', True)
    x = Variable(torch.randn(1,3,1000,800))
    net.cuda()
    x = x.cuda()
    pred = net(x)
    print("input size: ", x.size())
    print("output size:", pred.size())

    import IPython
    IPython.embed()
