# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:51:17 2022

@author: vpillai
"""

from PIL import Image, ImageOps
import os

def fetch_data(path):
  new_path = path + '_resized/'
  os.mkdir(new_path)
  for img_path in os.listdir(path): # iterate through subject folder in training data folder
      if (str(img_path)[0] == '.'): # avoid files starting with .
        continue
      image = Image.open(path + "/" + img_path)
      image = resize(image) # interpolate/resize/standardize image (using openCV)
      new_image_path = new_path + img_path
      image.save(new_image_path, format='JPEG', quality=100)
  return 

def resize(im):
  desired_size = 450
  old_size = im.size  # old_size[0] is in (width, height) format

  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  im = im.resize(new_size, Image.ANTIALIAS)
  
  # create a new image and paste the resized on it

  new_im = Image.new("RGB", (desired_size, desired_size))
  new_im.paste(im, ((desired_size-new_size[0])//2,
                      (desired_size-new_size[1])//2))
  return new_im
