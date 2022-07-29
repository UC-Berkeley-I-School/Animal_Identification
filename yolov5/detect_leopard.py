# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
import shutil
#from typing import Annotated
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)


import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import torch
import torch.backends.cudnn as cudnn
import numpy
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as torch_data
import torch

transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

from siamese_triplet.trainer import fit
import numpy as np
from siamese_triplet.datasets import BalancedBatchSampler
from siamese_triplet.datasets import LeopardDataset
import torch.nn as nn

# Set up the network and training parameters
from siamese_triplet.networks import EmbeddingNet
from siamese_triplet.networks import EmbeddingWithSoftmaxNet
from siamese_triplet.networks import MultiPartEmbeddingNet
from siamese_triplet.networks import MultiPartEmbeddingWithSoftmaxNet

from siamese_triplet.losses import OnlineTripletLoss
from siamese_triplet.losses import OnlineSymTripletLoss
from siamese_triplet.losses import OnlineModTripletLoss
from siamese_triplet.utils_triplet import AllTripletSelector
from siamese_triplet.utils_triplet import HardestNegativeTripletSelector
from siamese_triplet.utils_triplet import RandomNegativeTripletSelector
from siamese_triplet.utils_triplet import SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from siamese_triplet.metrics import AverageNonzeroTripletsMetric
from sklearn.metrics import f1_score, classification_report 

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        id_weights= ROOT / 'leop_id_model_july_24.pt', # weights for re-id/ood
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        flank_size=(64, 64), # size of resized leopard flank
        leop_size=(96, 128), # size of resized leopard 
        face_size=(64,64), # size of resized leopard head
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        resize=True, # resize leopard image
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        multi_embedding=False, # create predictions based on multi-embedding of face/flank/full
        softmax=True, # utilize softmax function to create prediction
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    total_ref_labels = []
    total_pred_labels = []
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det) and ((3 in reversed(det)[...,-1:] and len(names) == 4) or (0 in reversed(det)[...,-1:] and len(names) == 1)):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                leop_objs = np.array(['flank','face'])
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image

                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())


                        confidence_score = conf
                        class_index = cls
                        object_name = names[int(cls)]
                        print(save_path)
                        print('confidence score is ', conf )
                        print('bounding box is ', x1, y1, x2, y2)
                        print('class index is ', class_index)
                        print('detected object name is ', object_name)
                        original_img = im0
                        cropped_img = im0[y1:y2, x1:x2]

                        # extract leopard number and full leopard path
                        l_class = save_path.split("/")[-1].split("_")[1]
                        l_img = save_path.split("/")[-1]

                        if l_img[0] == 'l': 
                        # setting correct image size and class names 
                          if object_name == 'head':
                            class_name = 'face'
                            object_name = 'face'
                            size = face_size
                          elif object_name == 'leopard':
                            class_name = 'full'
                            object_name = 'leop' 
                            size = leop_size
                          else:
                            class_name = 'flank'
                            object_name = 'flank'
                            size = flank_size
                      
                        # keep track of which classes have been detected (flank/face/full)
                        leop_objs = np.delete(leop_objs, np.where(leop_objs == class_name))

                        # save cropped images in appropriate folder specified by 'project' input
                        cropped_folder = str(project) + '/cropped_images/leop_' + l_class + '/' + class_name + '/'                         
                        os.makedirs(cropped_folder, exist_ok=True)
                        cropped_path = cropped_folder + object_name + '_' + '_'.join(l_img.split('_')[1:])                        
                        cv2.imwrite(cropped_path,cropped_img) 
                        LOGGER.info(f'Cropped image saved at ' + cropped_path)
                        
                        # resize image based on class name in appropriate folder specified by 'project' input
                        if resize:
                            # open cropped image and find new size based on old size and class name
                            # keep aspect ratio constant
                            cr_img = Image.open(cropped_path)
                            scale_y, scale_x = cr_img.size  # old_size[0] is in (width, height) format

                            # Adjust the height/width to specified image_size

                            if scale_y > (size[0]) :
                              scale = scale_y/size[0]
                              scale_y = size[0]
                              scale_x =  int(scale_x*1.0/scale)
                                  
                                          
                            if  scale_x > size[1] :
                              scale = scale_x/size[1]
                              scale_x = size[1]
                              scale_y =  int(scale_y*1.0/scale)

                            new_size = (scale_y, scale_x)
                            im = cr_img.resize(new_size, Image.ANTIALIAS)
                            
                            # create a new image and paste the resized on it 
                            new_im = Image.new("RGB", (size[0], size[1]))
                            new_im.paste(im, ((size[0]-new_size[0])//2,
                                                (size[1]-new_size[1])//2))
                            resized_folder = str(project) + '/_resized/leop_' + l_class + '/' + class_name + '/'                         
                            os.makedirs(resized_folder, exist_ok=True)
                            resized_path = resized_folder + object_name + '_' + '_'.join(l_img.split('_')[1:])                        
                            new_im.save(resized_path, format='JPEG', quality=100)
                            LOGGER.info(f'Resized image saved at ' + resized_path)

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
                # create blank file for undetected classes in cropped/resized folders
                if len(leop_objs):
                  for cl in leop_objs:
                    cropped_folder = str(project) + '/cropped_images/leop_' + l_class + '/' + cl + '/'
                    resized_folder = str(project) + '/_resized/leop_' + l_class + '/' + cl + '/'  
                    if cl == 'face':
                      size = face_size
                      cropped_path = cropped_folder + 'face' + '_' + '_'.join(l_img.split('_')[1:])
                      resized_path = resized_folder + 'face' + '_' + '_'.join(l_img.split('_')[1:])
                    else:
                      size = flank_size
                      cropped_path = cropped_folder + 'flank' + '_' + '_'.join(l_img.split('_')[1:])
                      resized_path = resized_folder + 'flank' + '_' + '_'.join(l_img.split('_')[1:])  

                    os.makedirs(cropped_folder, exist_ok=True)
                    cv2.imwrite(cropped_path, np.zeros([size[0],size[1],3],dtype=np.uint8))                     
                    os.makedirs(resized_folder, exist_ok=True)
                    cv2.imwrite(resized_path, np.zeros([size[0],size[1],3],dtype=np.uint8))
                
                # .... #

                # IDENTIFICATION #

                # .... #

              

                # creating temp folders for predictions
                os.makedirs(str(project) + '/_resized/temp/' + 'leop_' + l_class + '/'+ 'flank', exist_ok = True)
                shutil.copyfile(str(project) + '/_resized/' + 'leop_' + l_class + '/'+ 'flank/flank_' + '_'.join(l_img.split('_')[1:]) 
                          , str(project) + '/_resized/temp/' + 'leop_' + l_class + '/'+ 'flank/flank_' + '_'.join(l_img.split('_')[1:]))
                os.makedirs(str(project) + '/_resized/temp/' + 'leop_' + l_class + '/'+ 'face', exist_ok = True)
                shutil.copyfile(str(project) + '/_resized/' + 'leop_' + l_class + '/'+ 'face/face_' + '_'.join(l_img.split('_')[1:]) 
                          , str(project) + '/_resized/temp/' + 'leop_' + l_class + '/'+ 'face/face_' + '_'.join(l_img.split('_')[1:]))
                os.makedirs(str(project) + '/_resized/temp/' + 'leop_' + l_class + '/'+ 'full', exist_ok = True)
                shutil.copyfile(str(project) + '/_resized/' + 'leop_' + l_class + '/'+ 'full/leop_' + '_'.join(l_img.split('_')[1:]) 
                          , str(project) + '/_resized/temp/' + 'leop_' + l_class + '/'+ 'full/leop_' + '_'.join(l_img.split('_')[1:]))


                cuda = torch.cuda.is_available()
                margin = 0.2
                TEST_DATA_PATH = str(project) + '/_resized/temp/'
                test_dataset = LeopardDataset(image_dir=TEST_DATA_PATH,transform=transform_img)
                
                if multi_embedding:
                    if softmax:
                        embedding_net = MultiPartEmbeddingWithSoftmaxNet(num_classes=64)
                    else:
                        embedding_net = MultiPartEmbeddingNet()
                else:    
                    if softmax:
                        embedding_net = EmbeddingWithSoftmaxNet(num_classes=64)
                    else:
                        embedding_net = EmbeddingNet()

                model_id = embedding_net

                if cuda:
                    model_id.cuda()
                loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
                lr = 1e-3
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
                n_epochs = 20
                log_interval = 50
                softmax_loss_fn = nn.CrossEntropyLoss()

                def extract_embeddings(dataloader, model, multi_class=False, softmax=False):
                    embeddings = []
                    ref_labels = []
                    pred_labels = []
                    with torch.no_grad():
                        model.eval()
                        
                        if multi_class:
                            for face, flank, full, target in dataloader:
                                if cuda:
                                    face = face.cuda()
                                    flank = flank.cuda()
                                    full = full.cuda()
                                if softmax:
                                    x,y=model.get_embedding(full) 
                                    z, preds = torch.max(y.data, 1)
                                    pred_labels.extend(preds.data.cpu().numpy().tolist())
                                else:
                                    x=model.get_embedding(full)
                                
                                embeddings.extend(x.data.cpu().numpy())
                                ref_labels.extend(target.data.cpu().numpy().tolist())
                        else:      
                            for face, flank, full, target in dataloader:
                                if cuda:
                                    full = full.cuda()
                                if softmax:
                                    x,y=model.get_embedding(full) 
                                    z, preds = torch.max(y.data, 1)
                                    pred_labels.extend(preds.data.cpu().numpy().tolist())
                                else:
                                    x=model.get_embedding(full)
                                
                                embeddings.extend(x.data.cpu().numpy())
                                ref_labels.extend(target.data.cpu().numpy().tolist())
                                
                    if softmax:        
                        return embeddings, ref_labels, pred_labels
                    else:
                        return embeddings, ref_labels

                model_id.load_state_dict(torch.load(id_weights[0]))

                test_eval_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False,  num_workers=2, drop_last=True, pin_memory=cuda)
                test_emb, test_ref_labels, test_pred_labels= extract_embeddings(test_eval_loader, model_id, multi_class=True, softmax=True)

                labels_dict =  {0: 'leop_196', 1: 'leop_190', 2: 'leop_277', 3: 'leop_75', 4: 'leop_86', 5: 'leop_26',
                6: 'leop_212', 7: 'leop_10', 8: 'leop_80', 9: 'leop_89', 10: 'leop_282', 11: 'leop_249', 12: 'leop_29',
                13: 'leop_18', 14: 'leop_34', 15: 'leop_94', 16: 'leop_7', 17: 'leop_291', 18: 'leop_56', 19: 'leop_0',
                20: 'leop_32', 21: 'leop_201', 22: 'leop_206', 23: 'leop_35', 24: 'leop_1', 25: 'leop_297', 26: 'leop_57', 
                27: 'leop_290', 28: 'leop_183', 29: 'leop_185', 30: 'leop_195', 31: 'leop_133', 32: 'leop_25', 33: 'leop_13',
                34: 'leop_227', 35: 'leop_14', 36: 'leop_40', 37: 'leop_280', 38: 'leop_274', 39: 'leop_78', 40: 'leop_82',
                41: 'leop_76', 42: 'leop_15', 43: 'leop_226', 44: 'leop_12', 45: 'leop_77', 46: 'leop_70', 47: 'leop_275', 
                48: 'leop_79', 49: 'leop_63', 50: 'leop_90', 51: 'leop_3', 52: 'leop_4', 53: 'leop_232', 54: 'leop_37', 
                55: 'leop_5', 56: 'leop_251', 57: 'leop_62', 58: 'leop_205', 59: 'leop_38', 60: 'leop_144', 61: 'leop_188',
                62: 'leop_186', 63: 'leop_121'}

                labels_inv_dict = dict(map(reversed, labels_dict.items()))
                test_pred_labels[0] = int(labels_dict[test_pred_labels[0]].split('_')[1])
                test_ref_labels[0] = int(test_dataset.n_classes[0].split('_')[1])
                
                if 'full' not in leop_objs: 
                  LOGGER.info(f'\nPredicted class of leopard ' + str(test_ref_labels[0]) 
                    +  ' is leopard ' + str(test_pred_labels[i]) + '.\n')

                total_ref_labels.append(test_ref_labels[0])
                total_pred_labels.append(test_pred_labels[0])
                shutil.rmtree(TEST_DATA_PATH)
            

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Creating prediction report
    print(classification_report(total_pred_labels, total_ref_labels))

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--id_weights', nargs='+', type=str, default=ROOT / 'leop_id_model_july_24.pt', help='id / ood weight path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--flank-size',nargs='+', type=int, default=[64,64], help='resized flank size h,w')
    parser.add_argument('--leop-size','--leopard-size', '--full-size',nargs='+', type=int, default=[64,64], help='resized leopard size h,w')
    parser.add_argument('--face-size', '--head-size', nargs='+', type=int, default=[96,128], help='resized leopard size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--resize', default=True, help='resize inference')
    parser.add_argument('--multi_embedding', default=False, help='resize inference')
    parser.add_argument('--softmax', default=True, help='resize inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
