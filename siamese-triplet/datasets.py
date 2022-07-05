import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

class LeopardDataset(Dataset):

    def __init__(self, image_dir=None, transform=None):
        self.image_dir = image_dir
        self.n_classes = os.listdir(image_dir)
        self.n_classes = [label for label in self.n_classes if label[0:4] == "leop"]
        self.labels_dict = {label: int(label.split('_')[-1]) for label in self.n_classes}
        self.labels_inv_dict = {int(label.split('_')[-1]) : label for label in self.n_classes}
        self.face_image_files = []
        self.flank_image_files = []
        self.full_image_files = []
        
        
        self.targets = []
        for label in self.n_classes:
            full_image_files = os.listdir(self.image_dir+'/'+label+'/full/')
            
            self.targets.extend(len(full_image_files)*[self.labels_dict[label]])
            self.face_image_files.extend(os.listdir(self.image_dir+'/'+label+'/face/'))
            self.flank_image_files.extend(os.listdir(self.image_dir+'/'+label+'/flank/'))    
            self.full_image_files.extend(full_image_files)   
                                         
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = self.labels_inv_dict[self.targets[index]]
        image_face = self.image_dir+'/'+label+'/face/'+self.face_image_files[index] 
        image_flank = self.image_dir+'/'+label+'/flank/'+self.flank_image_files[index] 
        image_full = self.image_dir+'/'+label+'/full/'+self.full_image_files[index] 
        image_face = PIL.Image.open(image_face)
        image_flank = PIL.Image.open(image_flank)
        image_full = PIL.Image.open(image_full)
        
        outputs = []
        if self.transform:
            outputs.append(torch.unsqueeze(self.transform(image_face),0))
            outputs.append(torch.unsqueeze(self.transform(image_flank),0))
            outputs.append(torch.unsqueeze(self.transform(image_full),0))
        out = torch.cat(outputs,dim=0)
        return out, self.targets[index]

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
