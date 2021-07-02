import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from glob import glob
from tqdm.notebook import tqdm
import cv2
from skimage import exposure
import pickle

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# config
dataset_dir = '../input/siim-covid19-detection'
class_names = ['Typical Appearance','Negative for Pneumonia', 
               'Indeterminate Appearance', 'Atypical Appearance']
label2color = {
    '[1, 0, 0, 0]': [255, 0, 0],    # Typical Appearance
    '[0, 1, 0, 0]': [0, 255, 0],    # Negative for Pneumonia
    '[0, 0, 1, 0]': [0, 0, 255],    # Indeterminate Appearance
    '[0, 0, 0, 1]': [255, 255, 0],  # Atypical Appearance
}

LABEL_CODE1 = {'[0 0 0 1]':0, '[0 0 1 0]':1, '[0 1 0 0]':2, '[1 0 0 0]':3}
LABEL_CODE2 = {0:'[0 0 0 1]', 1:'[0 0 1 0]', 2:'[0 1 0 0]', 3:'[1 0 0 0]'}
size = 300


# reading tabular data
train = pd.read_csv(f'{dataset_dir}/train_image_level.csv')
train_study = pd.read_csv(f'{dataset_dir}/train_study_level.csv')

train_study['StudyInstanceUID'] = train_study['id'].apply(lambda x: x.replace('_study', ''))
del train_study['id']
train = train.merge(train_study, on='StudyInstanceUID')

# reading DICOM images
def dicom2array(path, voi_lut=True, fix_monochrome=True):
#     dicom = pydicom.read_file(path)
    dicom = pydicom.dcmread(path)
    try:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    except RuntimeError:  
        # when a dicom image is compressed it throws a runtime error -> https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html#supported-transfer-syntaxes 
        # also this link: https://pydicom.github.io/pydicom/stable/tutorials/installation.html#install-the-optional-libraries
        data = np.zeros([dicom.Rows, dicom.Columns])
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def create_dataset(training_df, data_dir, data_type):
    '''
    params:
        training_df: training dataset that contains image ids
        data_dir: dataset dir
        data_type: 'train' or 'test'
    return:
        images and targets arrays
    '''
    assert data_type in ['train', 'test'], 'data_type can only get either of train or test'
    data_dicts = []
    # create empty list to store image vectors
    images = []
    # create empty list to store targets
    targets = []
    # look over the dataframe
    for index, row in tqdm(
        training_df.iterrows(), 
        total=len(training_df), 
        desc="processing images"
    ):
        record = {}
        # get image id
        image_id = row["StudyInstanceUID"]
        # create image path
#         image_path = os.path.join(image_dir, image_id)
        image_path = glob(f'{data_dir}/{data_type}/{image_id}/*/*')[0]
        # open image
        image = dicom2array(image_path)
        # resize image
        w, h = image.shape
        image = cv2.resize(image, dsize=(size, size))
        # histogram equalization to ehance contrast
        image = exposure.equalize_hist(image)
        # storing the image
        image = np.stack([image, image, image], axis=-1)  # make it a 3-channel image
        fn = f'./{data_type}/{image_id}.jpg'
        cv2.imwrite(fn, image)
        
        # processing bounding boxe
        bbox, bboxes = [], []
        
        for i, l in enumerate(row['label'].split(' ')):
            if l[0] == 'none':
                if (i%6 == 1) | (i%6 == 2) | (i%6 == 3) | (i%6 == 4):
                    bbox.append(0)
                if i%6 == 5:
                    bboxes.append(bbox)
                    bbox = []
            else:
                if (i%6 == 1) | (i%6 == 3):
                    l = float(l)
                    bbox.append(l/w*size)
                if (i%6 == 2) | (i%6 == 4):
                    l = float(l)
                    bbox.append(l/h*size)
                if i%6 == 5:
                    bbox.append(LABEL_CODE1[str(row[class_names].values)])
        
        record["file_name"] = fn
        record["image_id"] = image_id
        record["height"] = h
        record["width"] = w
        objs = []
        
        if bbox != []:
            h_ratio = size / h
            w_ratio = size / w
            bbox_resized = [
                float(bbox[0]) * w_ratio,
                float(bbox[1]) * h_ratio,
                float(bbox[2]) * w_ratio,
                float(bbox[3]) * h_ratio,
            ]
            obj = {
                "bbox": bbox_resized,
#                 "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": LABEL_CODE1[str(row[class_names].values)],
            }
        else:
            bbox_resized = [50, 50, 200, 200]
            obj = {
                "bbox": bbox_resized,
#                 "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": LABEL_CODE1[str(row[class_names].values)],
            }
        
        objs.append(obj)
        record["annotations"] = objs
        data_dicts.append(record)
    
    with open('train_data', mode="wb") as f:
        pickle.dump(data_dicts, f)
        
    return data_dicts  

