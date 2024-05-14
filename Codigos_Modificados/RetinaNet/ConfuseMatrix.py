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
from tqdm import tqdm


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
    
    if mod_pred == 'single' or mod_pred == 'total':
        VN=0
        VP=0
        TP=0
        TN=0
        FN=0
        FP=0
        for lb in labelsr:
            if lb == 'opacity':
                TP=TP+1
            else:
                TN=TN+1

    if mod_pred == 'Xray14':
        AA=0
        EA=0
        CA=0
        IA=0
        NfA=0
        NeA=0
        NtA=0
        MA=0
        NA=0
        AE=0
        EE=0
        CE=0
        IE=0
        NfE=0
        NeE=0
        NtE=0
        ME=0
        NE=0
        AC=0
        EC=0
        CC=0
        IC=0
        NfC=0
        NeC=0
        NtC=0
        MC=0
        NC=0
        AI=0
        EI=0
        CI=0
        II=0
        NfI=0
        NeI=0
        NtI=0
        MI=0
        NI=0
        ANf=0
        ENf=0
        CNf=0
        INf=0
        NfNf=0
        NeNf=0
        NtNf=0
        MNf=0
        NNf=0
        ANe=0
        ENe=0
        CNe=0
        INe=0
        NfNe=0
        NeNe=0
        NtNe=0
        MNe=0
        NNe=0
        ANt=0
        ENt=0
        CNt=0
        INt=0
        NfNt=0
        NeNt=0
        NtNt=0
        MNt=0
        NNt=0
        AM=0
        EM=0
        CM=0
        IM=0
        NfM=0
        NeM=0
        NtM=0
        MM=0
        NM=0
        AN=0
        EN=0
        CN=0
        IN=0
        NfN=0
        NeN=0
        NtN=0
        MN=0
        NN=0             

    if mod_pred == 'SIIM':
        TT=0
        TN=0
        TI=0
        TA=0
        NT=0
        NN=0
        NI=0
        NA=0
        IT=0
        IN=0
        II=0
        IA=0
        AT=0
        AN=0
        AI=0
        AA=0 

    for i,ruta in tqdm(enumerate(rutas)):
        
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

                clss= classification.cpu().numpy().tolist()
                clss=list(set(clss))

                if isinstance(true_label, str):
                    pass

                elif np.isnan(true_label):
                    true_label=''

                if len(clss) == 0 and true_label == '':
                    VN=VN+1
                if len(clss) == 1 and clss[0] == 0 and true_label == 'opacity':
                    VP=VP+1
            
            if mod_pred == 'total':
                pred=classification.cpu().numpy().tolist()
                sc=scores.cpu().numpy().tolist()
                idx=sc.index(max(sc))
                predmax=pred[idx]
                
                if predmax == 0 and true_label == 'opacity':
                    VP=VP+1
                if predmax == 1 and true_label == 'Noopacity':
                    VN=VN+1

            if mod_pred == 'Xray14':

                pred=classification.cpu().numpy().tolist()
                sc=scores.cpu().numpy().tolist()
                idx=sc.index(max(sc))
                predmax=pred[idx]
            

                if predmax == 0 and true_label == 'Atelectasis':
                    AA=AA+1
                if predmax == 0 and true_label == 'Effusion':
                    EA=EA+1
                if predmax == 0 and true_label == 'Cardiomegaly':
                    CA=CA+1
                if predmax == 0 and true_label == 'Infiltrate':
                    IA=IA+1 
                if predmax == 0 and true_label == 'No Finding':
                    NfA=NfA+1
                if predmax == 0 and true_label == 'Pneumonia':
                    NeA=NeA+1 
                if predmax == 0 and true_label == 'Pneumothorax':
                    NtA=NtA+1
                if predmax == 0 and true_label == 'Mass':
                    MA=MA+1 
                if predmax == 0 and true_label == 'Nodule':
                    NA=NA+1

                if predmax == 1 and true_label == 'Atelectasis':
                    AE=AE+1
                if predmax == 1 and true_label == 'Effusion':
                    EE=EE+1
                if predmax == 1 and true_label == 'Cardiomegaly':
                    CE=CE+1
                if predmax == 1 and true_label == 'Infiltrate':
                    IE=IE+1 
                if predmax == 1 and true_label == 'No Finding':
                    NfE=NfE+1
                if predmax == 1 and true_label == 'Pneumonia':
                    NeE=NeE+1 
                if predmax == 1 and true_label == 'Pneumothorax':
                    NtE=NtE+1
                if predmax == 1 and true_label == 'Mass':
                    ME=ME+1 
                if predmax == 1 and true_label == 'Nodule':
                    NE=NE+1

                if predmax == 2 and true_label == 'Atelectasis':
                    AC=AC+1
                if predmax == 2 and true_label == 'Effusion':
                    EC=EC+1
                if predmax == 2 and true_label == 'Cardiomegaly':
                    CC=CC+1
                if predmax == 2 and true_label == 'Infiltrate':
                    IC=IC+1 
                if predmax == 2 and true_label == 'No Finding':
                    NfC=NfC+1
                if predmax == 2 and true_label == 'Pneumonia':
                    NeC=NeC+1 
                if predmax == 2 and true_label == 'Pneumothorax':
                    NtC=NtC+1
                if predmax == 2 and true_label == 'Mass':
                    MC=MC+1 
                if predmax == 2 and true_label == 'Nodule':
                    NC=NC+1


                if predmax == 3 and true_label == 'Atelectasis':
                    AI=AI+1
                if predmax == 3 and true_label == 'Effusion':
                    EI=EI+1
                if predmax == 3 and true_label == 'Cardiomegaly':
                    CI=CI+1
                if predmax == 3 and true_label == 'Infiltrate':
                    II=II+1 
                if predmax == 3 and true_label == 'No Finding':
                    NfI=NfI+1
                if predmax == 3 and true_label == 'Pneumonia':
                    NeI=NeI+1 
                if predmax == 3 and true_label == 'Pneumothorax':
                    NtI=NtI+1
                if predmax == 3 and true_label == 'Mass':
                    MI=MI+1 
                if predmax == 3 and true_label == 'Nodule':
                    NI=NI+1


                if predmax == 4 and true_label == 'Atelectasis':
                    ANf=ANf+1
                if predmax == 4 and true_label == 'Effusion':
                    ENfENf+1
                if predmax == 4 and true_label == 'Cardiomegaly':
                    CNf=CNf+1
                if predmax == 4 and true_label == 'Infiltrate':
                    INf=INf+1 
                if predmax == 4 and true_label == 'No Finding':
                    NfNf=NfNf+1
                if predmax == 4 and true_label == 'Pneumonia':
                    NeNf=NeNf+1 
                if predmax == 4 and true_label == 'Pneumothorax':
                    NtNf=NtNf+1
                if predmax == 4 and true_label == 'Mass':
                    MNf=MNf+1 
                if predmax == 4 and true_label == 'Nodule':
                    NNf=NNf+1


                if predmax == 5 and true_label == 'Atelectasis':
                    ANe=ANe+1
                if predmax == 5 and true_label == 'Effusion':
                    ENe=ENe+1
                if predmax == 5 and true_label == 'Cardiomegaly':
                    CNe=CNe+1
                if predmax == 5 and true_label == 'Infiltrate':
                    INe=INe+1 
                if predmax == 5 and true_label == 'No Finding':
                    NfNe=NfNe+1
                if predmax == 5 and true_label == 'Pneumonia':
                    NeNe=NeNe+1 
                if predmax == 5 and true_label == 'Pneumothorax':
                    NtNe=NtNe+1
                if predmax == 5 and true_label == 'Mass':
                    MNe=MNe+1 
                if predmax == 5 and true_label == 'Nodule':
                    NNe=NNe+1


                if predmax == 6 and true_label == 'Atelectasis':
                    ANt=ANt+1
                if predmax == 6 and true_label == 'Effusion':
                    ENt=ENt+1
                if predmax == 6 and true_label == 'Cardiomegaly':
                    CNt=CNt+1
                if predmax == 6 and true_label == 'Infiltrate':
                    INt=INt+1 
                if predmax == 6 and true_label == 'No Finding':
                    NfNt=NfNt+1
                if predmax == 6 and true_label == 'Pneumonia':
                    NeNt=NeNt+1 
                if predmax == 6 and true_label == 'Pneumothorax':
                    NtNt=NtNt+1
                if predmax == 6 and true_label == 'Mass':
                    MNt=MNt+1 
                if predmax == 6 and true_label == 'Nodule':
                    NNt=NNt+1


                if predmax == 7 and true_label == 'Atelectasis':
                    AM=AM+1
                if predmax == 7 and true_label == 'Effusion':
                    EM=EM+1
                if predmax == 7 and true_label == 'Cardiomegaly':
                    CM=CM+1
                if predmax == 7 and true_label == 'Infiltrate':
                    IM=IM+1 
                if predmax == 7 and true_label == 'No Finding':
                    NfM=NfM+1
                if predmax == 7 and true_label == 'Pneumonia':
                    NeM=NeM+1 
                if predmax == 7 and true_label == 'Pneumothorax':
                    NtM=NtM+1
                if predmax == 7 and true_label == 'Mass':
                    MM=MM+1 
                if predmax == 7 and true_label == 'Nodule':
                    NM=NM+1


                if predmax == 8 and true_label == 'Atelectasis':
                    AN=AN+1
                if predmax == 8 and true_label == 'Effusion':
                    EN=EN+1
                if predmax == 8 and true_label == 'Cardiomegaly':
                    CN=CN+1
                if predmax == 8 and true_label == 'Infiltrate':
                    IN=IN+1 
                if predmax == 8 and true_label == 'No Finding':
                    NfN=NfN+1
                if predmax == 8 and true_label == 'Pneumonia':
                    NeN=NeN+1 
                if predmax == 8 and true_label == 'Pneumothorax':
                    NtN=NtN+1
                if predmax == 8 and true_label == 'Mass':
                    MN=MN+1 
                if predmax == 8 and true_label == 'Nodule':
                    NN=NN+1                        
                
            
            if mod_pred == 'SIIM':

                pred=classification.cpu().numpy().tolist()
                sc=scores.cpu().numpy().tolist()
                idx=sc.index(max(sc))
                predmax=pred[idx]
                


                if predmax == 0 and true_label == 'Typical Appearance':
                    TT=TT+1
                if predmax == 0 and true_label == 'Negative for Pneumonia':
                    TN=TN+1
                if predmax == 0 and true_label == 'Indeterminate Appearance':
                    TI=TI+1
                if predmax == 0 and true_label == 'Atypical Appearance':
                    TA=TA+1 
                

                if predmax == 1 and true_label == 'Typical Appearance':
                    NT=NT+1
                if predmax == 1 and true_label == 'Negative for Pneumonia':
                    NN=NN+1
                if predmax == 1 and true_label == 'Indeterminate Appearance':
                    NI=NI+1
                if predmax == 1 and true_label == 'Atypical Appearance':
                    NA=NA+1 
                

                if predmax == 2 and true_label == 'Typical Appearance':
                    IT=IT+1
                if predmax == 2 and true_label == 'Negative for Pneumonia':
                    IN=IN+1
                if predmax == 2 and true_label == 'Indeterminate Appearance':
                    II=II+1
                if predmax == 2 and true_label == 'Atypical Appearance':
                    IA=IA+1 
                


                if predmax == 3 and true_label == 'Typical Appearance':
                    AT=AT+1
                if predmax == 3 and true_label == 'Negative for Pneumonia':
                    AN=AN+1
                if predmax == 3 and true_label == 'Indeterminate Appearance':
                    AI=AI+1
                if predmax == 3 and true_label == 'Atypical Appearance':
                    AA=AA+1 
               


               

                        

    if mod_pred == 'single' or mod_pred == 'total':

        FN=TN-VN
        FP=TP-VP

        matriz_confusion = [[VP, FP],[FN, VN]]
    
        clases = ['Opacidad', 'Sanos']    
        sns.set(font_scale=1.2)  # Ajusta el tamaño de la fuente
        plt.figure(figsize=(10, 10))
        sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16},xticklabels=clases, yticklabels=clases)
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.title('')
    

    if mod_pred == 'Xray14':

        matriz_confusion = [[AA,EA,CA,IA,NfA,NeA,NtA,MA,NA],[AE,EE,CE,IE,NfE,NeE,NtE,ME,NE],[AC,EC,CC,IC,NfC,NeC,NtC,MC,NC],[AI,EI,CI,II,NfI,NeI,NtI,MI,NI],[ANf,ENf,CNf,INf,NfNf,NeNf,NtNf,MNf,NNf],[ANe,ENe,CNe,INe,NfNe,NeNe,NtNe,MNe,NNe],[ANt,ENt,CNt,INt,NfNt,NeN,NtNt,MNt,NNt],[AM,EM,CM,IM,NfM,NeM,NtM,MM,NM],[AN,EN,CN,IN,NfN,NeN,NtN,MN,NN]]
    
        clases = ['A', 'E','C','I','Nf','Ne','Nt','M','N']  
        sns.set(font_scale=12)  # Ajusta el tamaño de la fuente
        plt.figure(figsize=(50, 50))
        sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 60},xticklabels=clases, yticklabels=clases)
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.title('')
    

    if mod_pred == 'SIIM':

        matriz_confusion = [[TT,TN,TI,TA],[NT,NN,NI,NA],[IT,IN,II,IA],[AT,AN,AI,AA]]
    
        clases = ['Típico', 'Negativo','Indeterminado','Atípico']  
        sns.set(font_scale=12)  # Ajusta el tamaño de la fuente
        plt.figure(figsize=(50, 50))
        sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 60},xticklabels=clases, yticklabels=clases)
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.title('')
    


    plt.savefig(save_path+mode+'.png')
    
    print('Matrix created in '+save_path)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--test_csv', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument('--mod_pred', help='kind of image labeling it can be: single (opacity images with boundingbox and no opacity images without annotation), total(opacity images has entire image anotation), Xray14 (ChestXray14 dataset), SIIM (Covid dataset)',type=str, default='single')
    parser.add_argument('--save_path', help='path to save matrix image',type=str, default='/home/jair/COVID/retinanet/output/train/')
    parser.add_argument('--mode', help='dataset type used it can be: test or validation',type=str, default='test')

    parser = parser.parse_args()
    
    Generate_Matrix(parser.test_csv, parser.model_path, parser.class_list,parser.mod_pred,parser.save_path,parser.mode)
