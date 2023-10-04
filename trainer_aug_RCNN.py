import torchxrayvision as xrv
import skimage, torch, torchvision
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
from skimage import io, transform
import matplotlib.patches as patches
import random
import sys
import re
from data_aug.data_aug import *
from data_aug.bbox_util import *
from torchsummary import summary
from torchviz import make_dot
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

def verify_bb(bb1,bb2,im1,im2):
    bb=[]
    if len(bb2)>0:
        bb=bb2
        im=im2
    else:
        bb=bb1
        im=im1
    return im, bb

class CustomDataset(Dataset):
    def __init__(self, csv_path, target_size, normalizacion="[-1,1]", type_resize='interpolation', augmentation=None,bboxtipe='coco',dataroot=''):
        self.data = pd.read_csv(csv_path)
        self.norm=normalizacion
        self.target_size = target_size
        self.type_resize = type_resize
        self.augmentation = augmentation
        self.bboxtipe = bboxtipe
        self.dataroot = dataroot

    def __len__(self):
        return len(self.data)

    def resize_image(self, image):
        if  self.type_resize == 'interpolation':
          resized_image=skimage.transform.resize(image, self.target_size)
        else:
          original_height, original_width = image.shape
          target_height, target_width = self.target_size

          # Calcula las escalas de cambio de tamaño en ambas dimensisones
          scale_height = target_height / original_height
          scale_width = target_width / original_width

          # Elige la escala más pequeña para asegurarte de que la imagen se ajuste en el nuevo tamaño
          scale = min(scale_height, scale_width)

          # Calcula las nuevas dimensiones después del cambio de tamaño
          new_height = int(original_height * scale)
          new_width = int(original_width * scale)

          # Realiza el cambio de tamaño utilizando el método 'letterboxing'
          resized_image = np.zeros((target_height, target_width))
          top = (target_height - new_height) // 2
          left = (target_width - new_width) // 2
          resized_image[top:top + new_height, left:left + new_width] = cv2.resize(image, (new_width, new_height))

        return resized_image

    def resize_bbox(self, bbox, original_shape):
        # Redimensionar las coordenadas del bounding box para que se ajusten al nuevo tamañofrom tqdm import tqdm
        x, y, width, height = bbox
        new_width = width * self.target_size[1] / original_shape[1]
        new_height = height * self.target_size[0] / original_shape[0]
        new_x = x * self.target_size[1] / original_shape[1]
        new_y = y * self.target_size[0] / original_shape[0]

        return [new_x, new_y, new_width, new_height]

    def apply_aug(self,image,bbox):
        if self.augmentation:
            bboxr=np.array([[bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]])
            imgr = np.stack((image, image, image)).transpose(1, 2, 0)
            imgr1, bboxr1=RandomHorizontalFlip(0.5)(imgr, bboxr)
            imgr1, bboxr1=verify_bb(bboxr,bboxr1,imgr,imgr1)
            imgr2, bboxr2=RandomScale(0.1, diff = True)(imgr1,bboxr1)
            imgr2, bboxr2=verify_bb(bboxr1,bboxr2,imgr1,imgr2)
            imgr3, bboxr3=RandomTranslate(0.05, diff = True)(imgr2, bboxr2)
            imgr3, bboxr3=verify_bb(bboxr2,bboxr3,imgr2,imgr3)
            imgr4, bboxr4=RandomRotate(5)(imgr3, bboxr3)
            imgr4, bboxr4=verify_bb(bboxr3,bboxr4,imgr3,imgr4)
            imgrf, bboxrf=RandomShear(0.1)(imgr4, bboxr4)
            imgrf, bboxrf=verify_bb(bboxr4,bboxrf,imgr4,imgrf)
            img=np.array(imgrf[...,0]).astype(np.uint8)
            bbox=[float(bboxrf[0][0]),float(bboxrf[0][1]),float(bboxrf[0][2]-bboxrf[0][0]),float(bboxrf[0][3]-bboxrf[0][1])]
        else:
            img=image
            bbox=bbox
        return img , bbox


    def __getitem__(self, idx):
        img_ruta = self.dataroot+self.data['ruta'][idx]
        image = io.imread(img_ruta)
        # Extract bounding box coordinates
        x, y, width, height = self.data.iloc[idx, 1:5]

        image,bbox=self.apply_aug(image,[x,y,width,height])

        if self.norm == "[-1,1]":
          #normaliza a [-1,1]
          if image.dtype == np.uint8:  # Si la imagen es de 8 bits
              image = (image / 255.0) * 2.0 - 1.0
          elif image.dtype == np.uint16:  # Si la imagen es de 16 bits
              image = (image / 65535.0) * 2.0 - 1.0
          else:
              raise ValueError("Tipo de datos de imagen no compatible.")
        else:
          #Normaliza a [0,1]
          if image.dtype == np.uint8:  # Si la imagen es de 8 bits
            image = image / 255.0
          elif image.dtype == np.uint16:  # Si la imagen es de 16 bits
            image = image / 65535.0
          else:
              raise ValueError("Tipo de datos de imagen no compatible.")

        # Redimensionar la imagen y el bounding box
        image_r = self.resize_image(image)
        bbox_r = self.resize_bbox(bbox, image.shape)
        image_r= image_r[np.newaxis, ...]
        area = bbox_r[2]*bbox_r[3]
        if self.bboxtipe != 'coco':
          bbox_r=[bbox_r[0],bbox_r[1],bbox_r[0]+bbox_r[2],bbox_r[1]+bbox_r[3]]
        image_r = torch.tensor(image_r, dtype=torch.float32)
        bbox_r = torch.tensor([bbox_r], dtype=torch.float32)

        labels = torch.ones((1,), dtype=torch.int64)
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = bbox_r
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = torch.tensor([area])
        target["iscrowd"] = iscrowd
        return image_r, target

dataroot = "/home/jair/"
csv_path_traind = dataroot+"Anotaciones_estructuradas/deteccion_train.csv"
csv_path_testd = dataroot+"Anotaciones_estructuradas/deteccion_test.csv"

new_size=(300,300)
normalize="[0,1]"
batch_size=10 
type_resize="interpolation" # letterboxing or interpolation
trans=True
bboxtipe='xy' #coco or xy 

set_traind_t=CustomDataset(csv_path_traind,target_size=new_size,normalizacion=normalize,type_resize=type_resize,augmentation=trans,bboxtipe=bboxtipe,dataroot=dataroot)
dataloader_traind_t = DataLoader(set_traind_t, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

set_testd_t=CustomDataset(csv_path_testd,target_size=new_size,normalizacion=normalize,type_resize=type_resize,augmentation=trans,bboxtipe=bboxtipe,dataroot=dataroot)
dataloader_testd_t = DataLoader(set_testd_t, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

def create_model(num_classes,pretrain=True):

    if pretrain:
        # load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrain)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrain)
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def cargar_modelo_entrenado(ruta_modelo):
    modelo =create_model(num_classes=2,pretrain=False)
    state_dict= torch.load(ruta_modelo)
    state_dict_nuevo={}
    for key,value in state_dict.items():
        key_n=key.replace("module.","")
        state_dict_nuevo[key_n]=value
    modelo.load_state_dict(state_dict_nuevo)
    return modelo
    

def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

def create_csv(model,name):
    train_csv = pd.DataFrame(columns=['epoch', 'loss'])
    train_path='Modelos/'+model+'/train'+name+'.csv'
    train_csv.to_csv(train_path, index=False)
    
    train_csv = pd.DataFrame(columns=['epoch', 'loss'])
    validation_path='Modelos/'+model+'/validation'+name+'.csv'
    train_csv.to_csv(validation_path, index=False)
    return train_path, validation_path

def create_graphs(train_path,validation_path,title_graph,name):
    train = pd.read_csv(train_path)
    val = pd.read_csv(validation_path)

    x1 = train['epoch']
    y1 = train['loss']

    x2 = val['epoch']
    y2 = val['loss']

    plt.plot(x1, y1, marker='o', linestyle='-') 
    plt.title('train graph '+title_graph)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('Modelos/RCNN/train'+name+'.png')
    plt.close()

    plt.plot(x2, y2, marker='o', linestyle='-') 
    plt.title('validation graph '+title_graph)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('Modelos/RCNN/validation'+name+'.png')
    plt.close()

train_path, validation_path=create_csv("RCNN","_orig_aug")

tipo = "gpus" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("There is gpu available:", torch.cuda.is_available())
print("The number of gpu's availables:", torch.cuda.device_count())

# load a model pre-trained in oriinal
ruta_modelo_entrenado = "Modelos/RCNN/modelo_entrenado_orig.pth"
model = cargar_modelo_entrenado(ruta_modelo_entrenado)

# Verificar si hay múltiples GPUs disponibles y usar DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# define the optimizer
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

# initialize the Averager class
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
train_loss_list = []
val_loss_list = []

    
num_epochs = 40

print('Starting training with: ', tipo)
for epoch in tqdm(range(num_epochs)):
    model.train()

    
    train_loss_hist.reset()
    val_loss_hist.reset()  

    train_loss = train(dataloader_traind_t,model)
    val_loss = validate(dataloader_testd_t,model)
    
    torch.cuda.empty_cache()

    with open(train_path, mode='a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the epoch and avetage loss in the CSV
        csvwriter.writerow([epoch, train_loss_list[epoch]])

    with open(validation_path, mode='a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        #Write the epoch and avetage loss in the CSV
        csvwriter.writerow([epoch, val_loss_list[epoch]])

# Guardar el modelo entrenado al final del entrenamiento
torch.save(model.state_dict(),'Modelos/RCNN/modelo_entrenado_orig_aug.pth')

create_graphs(train_path,validation_path,'original data plus augmentation',"_orig_aug")
