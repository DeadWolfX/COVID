import argparse
import collections
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

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


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--save_path',help='path to save evaluation',type=str, default='/home/jair/COVID/retinanet/output/train')
    parser.add_argument('--num_save',help='epochs cycle of save the model',type=int, default=1)
    parser.add_argument('--optimizer',help='optimizer type',type=str, default='Adam')
    parser.add_argument('--patience',help='patience',type=int, default=30)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
   

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True
    
    if parser.optimizer == 'Adam':
        optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    if parser.optimizer == 'SGD':         
        optimizer = optim.SGD(retinanet.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay= 0.0001, nesterov=False)
    if parser.optimizer == 'AdamW':         
        optimizer = optim.AdamW(retinanet.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)    
    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()
    
    with open(parser.csv_classes, 'r') as f:
                classes = load_classes(csv.reader(f, delimiter=','))

    
    print('Num training images: {}'.format(len(dataset_train)))
    

    MAPSCT=np.array([])
    LOSST=np.array([])
    MmAP=0
    Mepoch=0
    Time=0
    for epoch_num in range(parser.epochs):
        if parser.patience != Time:
            retinanet.train()
            retinanet.module.freeze_bn()

            epoch_loss = []
            for iter_num, data in enumerate(dataloader_train):
                try:
                    optimizer.zero_grad()

                    if torch.cuda.is_available():
                        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                    else:
                        classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                        
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss

                    if bool(loss == 0):
                        continue

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                    optimizer.step()

                    loss_hist.append(float(loss))

                    epoch_loss.append(float(loss))

                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))


                    del classification_loss
                    del regression_loss
                except Exception as e:
                    print(e)
                    continue
            
          
            if parser.dataset == 'coco':

                print('Evaluating dataset')

                coco_eval.evaluate_coco(dataset_val, retinanet)


            
            elif parser.dataset == 'csv' and parser.csv_val is not None:

                print('Evaluating dataset')

                mAP = csv_eval.evaluate(dataset_val, retinanet,save_path=parser.save_path)
                
                S=0
                for i in range(len(classes)):
                    S=S+mAP[i][0]
                
                MAPSCT=np.append(MAPSCT,S/len(classes))     
            LOSST=np.append(LOSST,np.mean(epoch_loss))
            scheduler.step(np.mean(epoch_loss))
           
            MmAP=max(MmAP,MAPSCT[epoch_num])
            if MmAP == MAPSCT[epoch_num]:
                Time=0
                Mepoch=epoch_num

                if os.path.exists(parser.save_path+'/'+'retinanet_best.pt'):
                    os.remove(parser.save_path+'/'+'retinanet_best.pt')
                    torch.save(retinanet.module,parser.save_path+'/'+'retinanet_best.pt')
                else:     
                    torch.save(retinanet.module,parser.save_path+'/'+'retinanet_best.pt')
            else:
                Time=Time+1        
            if parser.num_save !=0: 
                if epoch_num % parser.num_save == 0: 
                    torch.save(retinanet.module, parser.save_path+'/'+'retinanet_{}.pt'.format(epoch_num))
        else:
            if parser.dataset == 'coco':

                print('Evaluating dataset')

                coco_eval.evaluate_coco(dataset_val, retinanet)


            
            elif parser.dataset == 'csv' and parser.csv_val is not None:

                print('Evaluating dataset')

                mAP = csv_eval.evaluate(dataset_val, retinanet,save_path=parser.save_path)
                
                S=0
                for i in range(len(classes)):
                    S=S+mAP[i][0]
                
                MAPSCT=np.append(MAPSCT,S/len(classes))     
            LOSST=np.append(LOSST,np.mean(epoch_loss))
            scheduler.step(np.mean(epoch_loss))
           
            MmAP=max(MmAP,MAPSCT[epoch_num])
            if MmAP == MAPSCT[epoch_num]:
                Mepoch=epoch_num
                if os.path.exists(parser.save_path+'/'+'retinanet_best.pt'):
                    os.remove(parser.save_path+'/'+'retinanet_best.pt')
                    torch.save(retinanet.module,parser.save_path+'/'+'retinanet_best.pt')
                else:     
                    torch.save(retinanet.module,parser.save_path+'/'+'retinanet_best.pt')
            if parser.num_save !=0: 
                if epoch_num % parser.num_save == 0: 
                    torch.save(retinanet.module, parser.save_path+'/'+'retinanet_{}.pt'.format(epoch_num))

            break        

    retinanet.eval()
    
    torch.save(retinanet, parser.save_path+'/model_final.pt')

    print('Best mAP optainend in epoch: ',Mepoch)
    print('Best mAP: ',MmAP)



    xs=list(range(len(MAPSCT+1))) 
    if len(MAPSCT)>0:
        plt.figure()
        plt.plot(xs,MAPSCT)
        plt.title('mAPCT')
        plt.savefig(parser.save_path+'/mAPCT.png')

    
    if len(LOSST)>0:
        plt.figure()
        plt.plot(xs,LOSST)
        plt.title('LOSST')
        plt.savefig(parser.save_path+'/LOSST.png')    
           


if __name__ == '__main__':
    main()




