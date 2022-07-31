import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torchvision
from torchvision import transforms
import torch.utils.data as data
import torchvision.models as model
from .datasets import LeopardDataset
from .utils_triplet import centdist

class InferenceNetwork:
    def __init__(self, 
                 infer_path=None, 
                 infer_labels=False, 
                 labels_dict=None,
                 model=None, 
                 ood_reject_thresh=0.5,
                 num_classes=64,
                 cuda=True):
        self.ood_reject_thresh = ood_reject_thresh
        self.model= model
        self.cuda = cuda
        self.infer_labels = infer_labels
        self.num_classes = num_classes
        transform_img = transforms.Compose([
           #transforms.Resize(size= (128, 128)),
            transforms.RandomAutocontrast(0.9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.infer_dataset = LeopardDataset(image_dir=infer_path,
                                            transform=transform_img,
                                            labels_from_files=infer_labels)
        
        self.labels_dict = labels_dict
        self.labels_inv_dict = {}
        for (key,value) in labels_dict.items():
            self.labels_inv_dict[value] = key
            
        self.labels_inv_dict[num_classes] = 'leop_UKN'    
       
        self.infer_loader = data.DataLoader(self.infer_dataset, 
                                  batch_size=1, 
                                  shuffle=False,  
                                  num_workers=2, 
                                  drop_last=True, 
                                  pin_memory=cuda)
        
    def run(self):
        infer_emb, infer_softmax, ref_labels, infer_pred_index= extract_embeddings(self.infer_loader, 
                                                                              self.model,
                                                                              multi_class=True, 
                                                                              softmax=True,
                                                                              labels_from_files=self.infer_labels,     
                                                                              cuda=self.cuda)
        
        
        
        centroids = self.model.centroids
        if self.cuda:
            centroids = centroids.cuda()

        infer_emb_dist = centdist(infer_emb, centroids)
        infer_emb_dist_sort, infer_dist_indices = torch.sort(infer_emb_dist, dim=1)
        infer_softmax_sort, infer_softmax_index = torch.sort(infer_softmax, dim=1)
        
        # OOD detection.  Set Uknown leopard index is 64
        for i in range(len(infer_pred_index)):
            if infer_emb_dist_sort[i,0] > self.ood_reject_thresh and infer_softmax_sort[i,-1] < 0.45 :
                infer_pred_index[i] = self.num_classes 
        
        infer_pred_labels = [self.labels_inv_dict[index] for index in infer_pred_index ]
        ref_labels = [ref_label if ref_label in self.labels_dict else 'leop_UKN' for ref_label in ref_labels]
        return ref_labels, infer_pred_labels
        

def extract_embeddings(dataloader,
                       model,
                       multi_class=False,
                       softmax=False,
                       labels_from_files = False,
                       cuda=False):
    embeddings = []
    softmax_out = []
    ref_labels = []
    pred_labels = []
    softmax_fn = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        model.eval()
        
        
        if multi_class:
            if labels_from_files:
                for face, flank, full, target, label in dataloader:
                    if cuda:
                        #face = face.cuda()
                        #flank = flank.cuda()
                        full = full.cuda()
                    if softmax:    
                        x,y=model.get_embedding(full)   
                        z, preds = torch.max(y.data, 1)
                        pred_labels.extend(preds.data.cpu().numpy().tolist())
                        softmax_out.append(softmax_fn(y))
                    else:
                        x=model.get_embedding(full)
                
                    embeddings.append(x)
                
                    ref_labels.extend(label) #target.data.cpu().numpy().tolist())
                
                
            else:
                for face, flank, full, target in dataloader:
                    if cuda:
                        #face = face.cuda()
                        #flank = flank.cuda()
                        full = full.cuda()
                    if softmax:    
                        x,y=model.get_embedding(full)   
                        z, preds = torch.max(y.data, 1)
                        pred_labels.extend(preds.data.cpu().numpy().tolist())
                        softmax_out.append(softmax_fn(y))
                    else:
                        x=model.get_embedding(full)
                
                    embeddings.append(x)
                
                    ref_labels.extend(target.data.cpu().numpy().tolist())
        else:      
            for data, target in dataloader:
                if cuda:
                    data = data.cuda()
                if softmax:    
                    x,y=model.get_embedding(data)   
                    z, preds = torch.max(y.data, 1)
                    pred_labels.extend(preds.data.cpu().numpy().tolist())
                    softmax_out.append(softmax_fn(y))
                else:
                    x=model.get_embedding(data)
                
                embeddings.append(x)
                ref_labels.extend(target.data.cpu().numpy().tolist())
                
    if softmax:   
        #return embeddings, softmax, ref_labels, pred_labels   
        return torch.cat(embeddings, dim=0), torch.cat(softmax_out, dim=0), ref_labels, pred_labels
    else:
        return torch.cat(embeddings, -1), ref_labels

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
    def __init__(self, num_classes=2, emb_size=512, dropout=0.3):
        super(EmbeddingWithSoftmaxNet, self).__init__()
        res = model.resnet18(pretrained=True)
        num_fc_features = res.fc.in_features
        res = list(res.children())[:-1]
        self.base = nn.Sequential(*res)
        self.linear_fc = nn.Linear(num_fc_features, emb_size)
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.dropout = dropout
        self.sum_centroids = torch.empty([self.num_classes,self.emb_size],dtype=torch.float32, requires_grad=False)
        self.cluster_size = torch.zeros([self.num_classes,1],dtype=torch.short)
        self.softmax_fc = nn.Linear(emb_size, num_classes)
        self.centroids = torch.empty([self.num_classes,self.emb_size],dtype=torch.float32, requires_grad=False)
        
    def forward(self, x):
        # shape [N, C]
        #x = F.avg_pool2d(x, x.size()[2:])
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,p=self.dropout)
        x = self.linear_fc(x)
        y = self.softmax_fc(x)
        x = F.normalize(x)
        return x, y
    
    def compute_centroids(self, x, target):
        
        target_array = target.cpu().numpy().tolist()
       
        label_index = []
        for i in range(self.num_classes):
            label_index.append([])
        for i, label in enumerate(target_array):
            label_index[label].append(i)

        for i, array in enumerate(label_index):
            self.centroids[i] = torch.sum(x[array], dim=0) 
            with torch.no_grad():
                self.sum_centroids[i] += self.centroids[i]
                self.cluster_size[i] += len(array)
            self.centroids[i] = self.centroids[i]/len(array)
    
        return self.centroids
    
    def reset_centroids(self):
        self.centroids.requires_grad = False
        self.sum_centroids.requires_grad = False
        return
        
    
    def compute_intra_dist_loss(self, x, target):
    
        target_array = target.cpu().numpy().tolist()
        #min_dist_centroids = self.centroids[self.compute_min_dist(x)]
        min_dist_centroids = self.centroids[target_array]
        dist = torch.sub(x,min_dist_centroids)
        dist = torch.sum(dist*dist)/len(target_array)
        return dist
    
    def compute_final_centroids(self):
        self.centroids = self.sum_centroids/self.cluster_size
        
    def compute_min_dist(self, x):
        min_dist_index = []
        for emb in x:
            dist = torch.sub(emb, self.centroids)
            dist = torch.sum(dist*dist, dim=1)
            d_index = torch.argmin(dist)
            min_dist_index.append(d_index.cpu().detach().item())
        return min_dist_index
    
    def compute_inter_dist_loss(self, target):
        losses = 0
        target_array = target.cpu().numpy().tolist()
        #min_dist_centroids = self.centroids[self.compute_min_dist(x)]
        min_dist_centroids = self.centroids[target_array]
        for centroids in min_dist_centroids:
            dist = torch.sub(centroids, self.centroids)
            dist = torch.sum(dist*dist, dim=1)
            values, indices = torch.sort(dist)
            losses = losses+values[1]
            return losses
        
    def get_embedding(self, x):
        return self.forward(x)    
    
  

class MultiPartEmbeddingNet(nn.Module):
    def __init__(self, face_emb_size=128, flank_emb_size=128, full_emb_size = 256):
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
        #x_f = F.dropout(x_f,p=0.4)
        outputs.append(self.linear_fc_face(x_f))
        
        x_f = self.flank_base(x_flank)
        x_f = x_f.view(x_f.size(0), -1)
        #x_f = F.dropout(x_f,p=0.4)
        outputs.append(self.linear_fc_flank(x_f))
        
        x_f = self.full_base(x_full)
        x_f = x_f.view(x_f.size(0), -1)
        #x_f = F.dropout(x_f,p=0.4)
        outputs.append(self.linear_fc_full(x_f))
        x = torch.cat(outputs, -1)
        x = F.dropout(x,p=0.4)
        x = F.normalize(x)
        return x

    def get_embedding(self, x_face, x_flank, x_full):
        return self.forward(x_face, x_flank, x_full)    

class MultiPartEmbeddingWithSoftmaxNet(nn.Module):
    def __init__(self, num_classes=2,face_emb_size=64, flank_emb_size=64, full_emb_size = 128):
        super(MultiPartEmbeddingWithSoftmaxNet, self).__init__()
        res = model.resnet18(pretrained=True)
        num_fc_features = res.fc.in_features
        self.num_classes = num_classes
        res = list(res.children())[:-1]
        self.face_base = nn.Sequential(*res)
        self.flank_base = nn.Sequential(*res)
        self.full_base = nn.Sequential(*res)
        self.linear_fc_face = nn.Linear(num_fc_features, face_emb_size)
        self.linear_fc_flank = nn.Linear(num_fc_features, flank_emb_size)
        self.linear_fc_full = nn.Linear(num_fc_features, full_emb_size)
        emb_size = face_emb_size + flank_emb_size + full_emb_size
        self.softmax_fc = nn.Linear(emb_size, num_classes)
        self.sum_centroids = torch.empty([num_classes,emb_size],dtype=torch.float32, requires_grad=False)
        self.cluster_size = torch.zeros([num_classes,1],dtype=torch.short)
        self.centroids = torch.empty([num_classes,emb_size],dtype=torch.float32, requires_grad=False)
        
    def forward(self, x_face, x_flank, x_full):
        # shape [N, C]
        #x = F.avg_pool2d(x, x.size()[2:])
        outputs = []
        x_f = self.face_base(x_face)
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.2)
        outputs.append(self.linear_fc_face(x_f))
        
        x_f = self.flank_base(x_flank)
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.2)
        outputs.append(self.linear_fc_flank(x_f))
        
        x_f = self.full_base(x_full)
        x_f = x_f.view(x_f.size(0), -1)
        x_f = F.dropout(x_f,p=0.2)
        outputs.append(self.linear_fc_full(x_f))
        x = torch.cat(outputs, -1)
        #x = F.dropout(x,p=0.3)
        y = self.softmax_fc(x)
        x = F.normalize(x)
        return x, y
    
    def compute_centroids(self, x, target):
        target_array = target.cpu().numpy().tolist()
       
        label_index = []
        for i in range(self.num_classes):
            label_index.append([])
        for i, label in enumerate(target_array):
            label_index[label].append(i)

        for i, array in enumerate(label_index):
            self.centroids[i] = torch.sum(x[array], dim=0) 
            self.sum_centroids[i] += self.centroids[i]
            self.cluster_size[i] += len(array)
            self.centroids[i] = self.centroids[i]/len(array)
    
        return self.centroids
    
    def compute_final_centroids(self):
        self.centroids = self.sum_centroids/self.cluster_size
        
    def reset_centroids(self):
        self.centroids.requires_grad = False
        self.sum_centroids.requires_grad = False
        return     
    
    def get_embedding(self, x_face, x_flank, x_full):
        return self.forward(x_face, x_flank, x_full)    
   
