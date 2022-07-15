#import fiftyone as fo
import numpy as np
import os
import shutil, sys
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import glob


# Creates YOLOV4 dataset
# Data split into train and test 
# train set in output_folder/data/obj and test in output_folder/images/test
# train files in output_folder/data/train.txt and test files in output_folder/data/test.txt

def create_yolov4_dataset(dataset, train_test_split = 0.8, output_folder='./'):
    anno = [0, 0, 0, 0, 0]
    train_files = []
    test_files = []
       
    np.random.seed(42)
    
    train_image_path =  output_folder+'/data/obj'                   
    isExist = os.path.exists(train_image_path)                  
                    
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(train_image_path)    
                        
    test_image_path =  output_folder+'/images/test'                       
    isExist = os.path.exists(test_image_path)

    if not isExist:
        os.makedirs(test_image_path)                    

    for sample in dataset:  
        anno[1:] = sample['ground_truth']['detections'][0]['bounding_box']
        height = sample['metadata']['height']
        width = sample['metadata']['width']
        
        
        if anno[3] >=1.0:
            anno[3] = 1-1.0/width
            
        if anno[4] >=1.0:
            anno[4] = 1-1.0/height    

        anno[1] = anno[1]+0.5*anno[3]
        anno[2] = anno[2]+0.5*anno[4]       
       
       
            
        old_image_file = sample['filepath']
        train_image_file = 'data/obj/' + old_image_file.split("/")[-1]
        new_image_file = output_folder + '/images/test/' + old_image_file.split("/")[-1]                 
        
        if(np.random.random_sample() > train_test_split):        
            test_files.append(new_image_file+'\n')
        else:
            train_files.append(train_image_file+'\n')
            new_image_file =  output_folder+'/'+ train_image_file    
                        
        new_anno_file = new_image_file.replace("jpg", "txt")              
           
        copy_str = 'cp ' + old_image_file + ' ' + new_image_file
        os.system(copy_str)
        
        with open(new_anno_file, 'w') as fp:
            for item in anno:
            # write each item on a new line
                fp.write("%s " % item)
            fp.write("\n")  
    
   
    with open(output_folder+'/data/test.txt', 'w') as fp:
        fp.writelines(test_files)         
    with open(output_folder+'/data/train.txt', 'w') as fp:
        fp.writelines(train_files)  
        
        
# Creates YOLOV5 dataset
# Data split into train and test 
# train set in output_folder/train/ and test in output_folder/test


def create_yolov5_dataset(dataset, train_test_split = 0.8, output_folder='./'):
    anno = [0, 0, 0, 0, 0]
    train_files = []
    test_files = []   
    np.random.seed(42)
    leopard_dict = {}
    class_count = 0
    train_image_path =  output_folder+'/train'                   
    isExist = os.path.exists(train_image_path)                  
                    
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(train_image_path+'/images') 
        os.makedirs(train_image_path+'/labels')  
                        
    test_image_path =  output_folder+'/test'                       
    isExist = os.path.exists(test_image_path)

    if not isExist:
        os.makedirs(test_image_path+'/images') 
        os.makedirs(test_image_path+'/labels')                    

    for sample in dataset:  
        anno[1:] = sample['ground_truth']['detections'][0]['bounding_box']
        name = sample['ground_truth']['detections'][0]['name']
        
        if name not in leopard_dict.keys():
            leopard_dict[name] = 'leop_'+str(class_count)+'_'
            class_count = class_count + 1
            
        file_suffix = leopard_dict[name]
        height = sample['metadata']['height']
        width = sample['metadata']['width']
        
       
        if anno[3] >=1.0:
            anno[3] = 1-1.0/width
            
        if anno[4] >=1.0:
            anno[4] = 1-1.0/height    

        anno[1] = anno[1]+0.5*anno[3]
        anno[2] = anno[2]+0.5*anno[4]       
     
        old_image_file = sample['filepath']
        new_file_name =  old_image_file.split("/")[-1][-10:]
        new_file_name = file_suffix + new_file_name
        if(np.random.random_sample() < train_test_split):    
            
            new_image_file = train_image_path + '/images/'+ new_file_name
            new_anno_file = train_image_path + '/labels/'+ new_file_name
                
        else:
            new_image_file =  test_image_path + '/images/'+ new_file_name     
            new_anno_file = test_image_path + '/labels/'+ new_file_name
                        
        new_anno_file = new_anno_file.replace("jpg", "txt")              
           
        copy_str = 'cp ' + old_image_file + ' ' + new_image_file
        os.system(copy_str)
        
        with open(new_anno_file, 'w') as fp:
            for item in anno:
            # write each item on a new line
                fp.write("%s " % item)
            fp.write("\n")  

def split_dev_test(dataPath, dev_test_split=0.5):
    '''
    Split test into dev and test with 0.5 split
    Input: data path to the leopard folder containing train and test folders
    Output: a dev directory with 50% of the test set
    Notes: The split is 50% by default and can be changes as needed
    '''
    np.random.seed(42)
    output_folder=dataPath+'/dev'   

    # If dev folder doesn't exist, create one
    isExist = os.path.exists(output_folder) 
    if not isExist:
      os.makedirs(output_folder+'/images') 
      os.makedirs(output_folder+'/labels') 
    
    # Attach to absolute path test and dev folders
    test_images_path = dataPath+'/test/images'
    test_label_path = dataPath+'/test/labels'

    dev_images_path = dataPath+'/dev/images'
    dev_label_path = dataPath+'/dev/labels'
    
    # given the threshold 0.5, split img and txt
    for img in os.listdir(test_images_path):
      if np.random.random_sample() < dev_test_split:
        shutil.move(test_images_path+'/'+img, dev_images_path)

        text_file=img[:-3]+'txt'
        shutil.move(test_label_path+'/'+text_file, dev_label_path)

def merge_dev_test(dataPath):
    test_images_path = dataPath+'/test/images'
    test_label_path = dataPath+'/test/labels'

    dev_images_path = dataPath+'/dev/images'
    dev_label_path = dataPath+'/dev/labels'

    for img in os.listdir(dev_images_path):
        shutil.move(dev_images_path+'/'+img, test_images_path)

        text_file=img[:-3]+'txt'
        shutil.move(dev_label_path+'/'+text_file, test_label_path)


def get_scale_image(image_file, image_size = (32,32)):
    image = imread(image_file)

                       
    height = image_size[0]
    width = image_size[1]
    scale_y = image.shape[0]
    scale_x = image.shape[1]
    
    # Adjust the height/width to specified image_size
    
    if scale_y > (height) :
        scale = scale_y/height
        scale_y = height
        scale_x =  int(scale_x*1.0/scale)
            
                    
    if  scale_x > width :
        scale = scale_x/width
        scale_x = width
        scale_y =  int(scale_y*1.0/scale)
     
    image = resize(image, (scale_y, scale_x))
    image = image*255
    image = image.astype('uint8') 
    mode = 'constant'
    y_pad = (height-scale_y)//2
    x_pad = (width-scale_x)//2
    if len(image.shape) > 2:
        image = np.pad(image, pad_width=[(y_pad, height-scale_y-y_pad),(x_pad, width-scale_x-x_pad),(0,0)], mode=mode)
    else:
        image = np.pad(image, pad_width=[(y_pad, height-scale_y-y_pad),(x_pad, width-scale_x-x_pad)], mode=mode)
            
    return(image)


def resize_crop_images(in_path=None, out_path=None, face_size=(64,64), 
                       flank_size=(64,64), full_size=(96,128)):
    
    labels = glob.glob(in_path+'/*')  
    
    for label in labels:
        faces = glob.glob(label+'/face/*')
        label_name = label.split('/')[-1]
        
        isExist = os.path.exists(out_path+'/'+label_name+'/face/')                  
        # Create a new directory because it does not exist     
        if not isExist:
           os.makedirs(out_path+'/'+label_name+'/face/') 
           os.makedirs(out_path+'/'+label_name+'/flank/') 
           os.makedirs(out_path+'/'+label_name+'/full/') 
        
        face_labels = [face.split('/')[-1] for face in faces]
        faces = [face for i, face in enumerate(faces) if face_labels[i][0:4] == 'face']
        face_images = [get_scale_image(face, image_size=face_size) for face in faces]
        resize_face_files = [out_path+'/'+label_name+'/face/'+face_label for face_label in face_labels]
        for i in range(len(resize_face_files)):
            imsave(resize_face_files[i], face_images[i])
    
        flanks = glob.glob(label+'/flank/*')
        flank_labels = [flank.split('/')[-1] for flank in flanks]
        flanks = [flank for i, flank in enumerate(flanks) if flank_labels[i][0:5] == 'flank']
        flank_images = [get_scale_image(flank, image_size=flank_size) for flank in flanks]
        resize_flank_files = [out_path+'/'+label_name+'/flank/'+flank_label for flank_label in flank_labels]
        for i in range(len(resize_flank_files)):
            imsave(resize_flank_files[i], flank_images[i])
    
        fulls = glob.glob(label+'/full/*')
        full_labels = [full.split('/')[-1] for full in fulls]
        fulls = [full for i, full in enumerate(fulls) if full_labels[i][0:4] == 'leop']

        full_images = [get_scale_image(full, image_size=full_size) for full in fulls]
        resize_full_files = [out_path+'/'+label_name+'/full/'+full_label for full_label in full_labels]
        for i in range(len(resize_full_files)):
            imsave(resize_full_files[i], full_images[i])
        
    return


def verify_part_images(in_path=None):
    
    labels = glob.glob(in_path+'/*')
    for label in labels:
        faces = glob.glob(label+'/face/*')
        faces = [face.split('/')[-1] for face in faces]
        faces = [face for face in faces if face[0:4] == 'face']
        faces = [face.split('_')[-1] for face in faces]
        faces.sort()
        
        flanks = glob.glob(label+'/flank/*')
        flanks = [flank.split('/')[-1] for flank in flanks]
        flanks = [flank  for flank in flanks if flank[0:5] == 'flank']
        flanks = [flank.split('_')[-1] for flank in flanks]
        flanks.sort()
        
        fulls = glob.glob(label+'/full/*')
        fulls = [full.split('/')[-1] for full in fulls]
        fulls = [full for full in fulls if full[0:4] == 'leop' ]
        fulls = [full.split('_')[-1] for full in fulls]
        fulls.sort()
        
        count_faces = len(faces)
        count_flanks = len(flanks)
        count_fulls = len(fulls)
        image_mismatch = False
        if (count_faces == count_flanks) and (count_faces == count_fulls) and (count_flanks == count_fulls):
            
            for i in range(len(fulls)):
                if (faces[i] != flanks[i]) or (faces[i] != fulls[i]) or (fulls[i] != flanks[i]):
                    image_mismatch = True
                    break
            if  image_mismatch:
                print("Mismatch in part images")
            else:    
                print("Part images match!!")      
                    
        else:        
            #print(label)
            #print(faces)
            #print(flanks)
            #print(fulls)
            print("Mismatch in part images count")
            print("Face:{}, Flank:{}, Full={}".format(count_faces, count_flanks, count_fulls))
    return        
