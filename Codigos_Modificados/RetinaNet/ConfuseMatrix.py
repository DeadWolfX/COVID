import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result




#genera matriz de confusión  
def Generate_Matrix(test_csv, model_path, class_list,mod_pred,save_path,mode):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    info=pd.read_csv(test_csv)
    rutas=list(info['rutas'])
    labelsr=list(info['label'])
    VN=0
    VP=0
    TP=0
    TN=0
    FN=0
    FP=0
    if mod_pred == 'single':
        for lb in labelsr:
            if lb == 'opacity':
                TP=TP+1
            else:
                TN=TN+1    
    for i,ruta in enumerate(rutas):
        
        img_name=ruta.split('/')[-1]
        image_path=ruta.replace(img_name,"")

        

        image = cv2.imread(os.path.join(image_path, img_name))
        
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        
        
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            
            scores, classification, transformed_anchors = model(image.cuda().float())
            
            true_label=labelsr[i]


            if mod_pred == 'single':

                if isinstance(true_label, str):
                    pass

                elif np.isnan(true_label):
                    true_label=''

                clss= classification.cpu().numpy().tolist()
                clss=list(set(clss))

                if len(clss) == 0 and true_label == '':
                    VN=VN+1
                if len(clss) == 1 and clss[0] == 0 and true_label == 'opacity':
                    VP=VP+1

                    
                    




  
    
    FN=TN-VN
    FP=TP-VP

    matriz_confusion = [[VP, FP],
                    [FN, VN]]
    
    clases = ['Opacidad', 'Sanos']   
    sns.set(font_scale=1.2)  # Ajusta el tamaño de la fuente
    plt.figure(figsize=(10, 10))
    sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16},xticklabels=clases, yticklabels=clases)
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('')

    # Mostrar el gráfico
    plt.savefig(save_path+mode+'.png')
    
    print('Matrix created in '+save_path)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--test_csv', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument('--mod_pred', help='kind of image labeling it can be: single (opacity images with boundingbox and no opacity images without annotation) ',type=str, default='single')
    parser.add_argument('--save_path', help='path to save matrix image',type=str, default='/home/jair/COVID/retinanet/output/train/')
    parser.add_argument('--mode', help='dataset type used it can be: test or validation',type=str, default='test')

    parser = parser.parse_args()
    
    Generate_Matrix(parser.test_csv, parser.model_path, parser.class_list,parser.mod_pred,parser.save_path,parser.mode)
