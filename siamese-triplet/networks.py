import torch.nn as nn
import torch.nn.functional as F
import math


import torchvision.models as model

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        res = model.resnet18(pretrained=True)
        res = list(res.children())[:-1]
        self.base = nn.Sequential(*res)
        self.linear_fc = nn.Linear(512, 512)
        
    def forward(self, x):
        # shape [N, C]
        #x = F.avg_pool2d(x, x.size()[2:])
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,p=0.4)
        x = self.linear_fc(x)
        x = F.normalize(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class MultiEmbeddingNet(nn.Module):
    def __init__(self):
        super(MultiEmbeddingNet, self).__init__()
        res = model.resnet18(pretrained=True)
        res = list(res.children())[:-1]
        self.base = nn.Sequential(*res)
        self.linear_fc_face = nn.Linear(512, 128)
        self.linear_fc_flank = nn.Linear(512, 128)
        self.linear_fc_full = nn.Linear(512, 128)
        
    def forward(self, x):
        # shape [N, C]
        #x = F.avg_pool2d(x, x.size()[2:])
        outputs = []
        x_f = self.base(x[:,0,:,:,:])
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.4)
        outputs.append(self.linear_fc_face(x_f))
        
        x_f = self.base(x[:,1,:,:,:])
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.4)
        outputs.append(self.linear_fc_flank(x_f))
        
        x_f = self.base(x[:,2,:,:,:])
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.4)
        outputs.append(self.linear_fc_full(x_f))
        x = torch.cat(outputs, -1)
        x = F.normalize(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)    