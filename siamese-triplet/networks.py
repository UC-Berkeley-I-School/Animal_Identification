import torch
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
    
    
class EmbeddingWithSoftmaxNet(nn.Module):
    def __init__(self, num_classes=2, emb_size=512):
        super(EmbeddingWithSoftmaxNet, self).__init__()
        res = model.resnet18(pretrained=True)
        num_fc_features = res.fc.in_features
        res = list(res.children())[:-1]
        self.base = nn.Sequential(*res)
        self.linear_fc = nn.Linear(num_fc_features, emb_size)
        self.num_classes = num_classes
        self.softmax_fc = nn.Linear(emb_size, num_classes)
        
    def forward(self, x):
        # shape [N, C]
        #x = F.avg_pool2d(x, x.size()[2:])
        x = self.base(x)
        x = x.view(x.size(0), -1)
        #x = F.dropout(x,p=0.4)
        x = self.linear_fc(x)
        y = self.softmax_fc(x)
        x = F.normalize(x)
        return x, y

    def get_embedding(self, x):
        return self.forward(x)
    

class MultiPartEmbeddingNet(nn.Module):
    def __init__(self, face_emb_size=64, flank_emb_size=64, full_emb_size = 128):
        super(MultiPartEmbeddingNet, self).__init__()
        res = model.resnet18(pretrained=True)
        num_fc_features = res.fc.in_features
        res = list(res.children())[:-1]
        self.face_base = nn.Sequential(*res)
        self.flank_base = nn.Sequential(*res)
        self.full_base = nn.Sequential(*res)
        self.linear_fc_face = nn.Linear(num_fc_features, face_emb_size)
        self.linear_fc_flank = nn.Linear(num_fc_features, flank_emb_size)
        self.linear_fc_full = nn.Linear(num_fc_features, full_emb_size)
        
    def forward(self, x_face, x_flank, x_full):
        # shape [N, C]
        #x = F.avg_pool2d(x, x.size()[2:])
        outputs = []
        x_f = self.face_base(x_face)
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.1)
        outputs.append(self.linear_fc_face(x_f))
        
        x_f = self.flank_base(x_flank)
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.1)
        outputs.append(self.linear_fc_flank(x_f))
        
        x_f = self.full_base(x_full)
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.1)
        outputs.append(self.linear_fc_full(x_f))
        x = torch.cat(outputs, -1)
        x = F.normalize(x)
        return x

    def get_embedding(self, x_face, x_flank, x_full):
        return self.forward(x_face, x_flank, x_full)    

class MultiPartEmbeddingWithSoftmaxNet(nn.Module):
    def __init__(self, num_classes=2,face_emb_size=128, flank_emb_size=128, full_emb_size = 128):
        super(MultiPartEmbeddingWithSoftmaxNet, self).__init__()
        res = model.resnet18(pretrained=True)
        num_fc_features = res.fc.in_features
        
        
        res = list(res.children())[:-1]
        self.face_base = nn.Sequential(*res)
        self.flank_base = nn.Sequential(*res)
        self.full_base = nn.Sequential(*res)
        self.linear_fc_face = nn.Linear(num_fc_features, face_emb_size)
        self.linear_fc_flank = nn.Linear(num_fc_features, flank_emb_size)
        self.linear_fc_full = nn.Linear(num_fc_features, full_emb_size)
        
    def forward(self, x):
        # shape [N, C]
        #x = F.avg_pool2d(x, x.size()[2:])
        outputs = []
        x_face = self.face_base(x[:,0,:,:,:])
        x_face = x_face.view(x_face.size(0), -1)
        x_f = F.dropout(x_face,p=0.2)
        outputs.append(self.linear_fc_face(x_f))
        
        x_flank = self.flank_base(x[:,1,:,:,:])
        x_flank = x_flank.view(x_flank.size(0), -1)
        x_f = F.dropout(x_flank,p=0.2)
        outputs.append(self.linear_fc_flank(x_f))
        
        x_full = self.full_base(x[:,2,:,:,:])
        x_full = x_full.view(x_full.size(0), -1)
        x_f = F.dropout(x_full,p=0.2)
        outputs.append(self.linear_fc_full(x_f))
        x = torch.cat(outputs, -1)
        x = F.normalize(x)
        return x
