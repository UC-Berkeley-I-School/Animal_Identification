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
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
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
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                leop_objs = np.array(['flank','face','full'])
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        # c = int(cls)  # integer class
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

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
                              old_size = cr_img.size  # old_size[0] is in (width, height) format
                              ratio = float(max(size))/max(old_size)
                              new_size = tuple([int(x*ratio) for x in old_size])

                              # resize image based on Pillow's anti-alias methodology
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
                if len(leop_objs) < 3:
                  for cl in leop_objs:
                    if cl == 'full':
                      size = leop_size
                      cropped_folder = str(project) + '/cropped_images/leop_' + l_class + '/' + cl + '/'
                      os.makedirs(cropped_folder, exist_ok=True)
                      cropped_path = cropped_folder + 'leop' + '_' + '_'.join(l_img.split('_')[1:])
                      cv2.imwrite(cropped_path, np.zeros([size[0],size[1],3],dtype=np.uint8))
                      resized_folder = str(project) + '/_resized/leop_' + l_class + '/' + cl + '/'                       
                      os.makedirs(resized_folder, exist_ok=True)
                      resized_path = resized_folder + 'leop' + '_' + '_'.join(l_img.split('_')[1:]) 
                      cv2.imwrite(resized_path, np.zeros([size[0],size[1],3],dtype=np.uint8))
                    elif cl == 'face':
                      size = face_size
                      cropped_folder = str(project) + '/cropped_images/leop_' + l_class + '/' + cl + '/'
                      os.makedirs(cropped_folder, exist_ok=True)
                      cropped_path = cropped_folder + 'face' + '_' + '_'.join(l_img.split('_')[1:])                        
                      cv2.imwrite(cropped_path, np.zeros([size[0],size[1],3],dtype=np.uint8))
                      resized_folder = str(project) + '/_resized/leop_' + l_class + '/' + cl + '/'                       
                      os.makedirs(resized_folder, exist_ok=True)
                      resized_path = resized_folder + 'face' + '_' + '_'.join(l_img.split('_')[1:]) 
                      cv2.imwrite(resized_path, np.zeros([size[0],size[1],3],dtype=np.uint8))
                    else:
                      size = flank_size
                      cropped_folder = str(project) + '/cropped_images/leop_' + l_class + '/' + cl + '/'
                      os.makedirs(cropped_folder, exist_ok=True)
                      cropped_path = cropped_folder + 'flank' + '_' + '_'.join(l_img.split('_')[1:])                        
                      cv2.imwrite(cropped_path, np.zeros([size[0],size[1],3],dtype=np.uint8))
                      resized_folder = str(project) + '/_resized/leop_' + l_class + '/' + cl + '/'                       
                      os.makedirs(resized_folder, exist_ok=True)
                      resized_path = resized_folder + 'flank' + '_' + '_'.join(l_img.split('_')[1:]) 
                      cv2.imwrite(resized_path, np.zeros([size[0],size[1],3],dtype=np.uint8))

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
