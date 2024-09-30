<h2>Project description</h2>
<h3>Here are the codes used in the development of the master's thesis titled "Deep Learning for the Identification and Localization of COVID-19 Induced Damage in Chest X-Rays." The folder structure in this main folder is as follows:</h3>
<ul>
    <li><b><a href="https://github.com/JairMathAI/COVID/tree/main/Anotaciones">Anotaciones 📂 :</a></b></li> Folder containing the various annotations used to train deep learning models. For each dataset, there is a folder containing annotations exclusively for <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a>, as the annotations for <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a> are generated from the files in the <a href="https://github.com/JairMathAI/COVID/tree/main/Datos">Datos</a> folder and the codes in the <a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Datos">Codigos_Datos:</a> folder, both present in this repository. 
    <li><b><a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Datos">Codigos_Datos 📂 :</a></b></li> Folder containing three Jupyter Lab notebooks, each with codes and functions used to generate the annotations contained in the Annotations folder and data processing for each set.
    <li><b><a href="https://github.com/JairMathAI/COVID/tree/main/Datos">Datos 📂 :</a></b></li> Contains text files for each dataset used, summarizing relevant information about each dataset in a structured manner.
    <li><b><a href="https://github.com/JairMathAI/COVID/tree/main/Exploratorios">Exploratorios 📂 :</a></b></li> Contains three Jupyter Lab notebooks, one for each dataset used. These notebooks contain the codes and functions used to perform the reported exploratory analyses.
    <li><b><a href="https://github.com/JairMathAI/COVID/blob/main/env.yml">env.yml 📄 :</a></b></li> Anaconda virtual environment configuration containing all the necessary libraries for exploratory analysis, visualization, data preprocessing, and label manipulation.
    <li><b><a href="https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link">Trainings:</a></b></li> The weights of the trained models described in the thesis, as well as the complete training information and graphics, are available in the following <a href="https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link">Google Drive folder 📂 </a>
    <li><b><a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Modificados">Codigos_Modificados 📂 :</a></b></li>Contains two folders, one for each model:
    <ul>
        <li><b>For <a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Modificados/YOLO">Y.O.L.O 📂 </a> </b></li> the code with modifications in the attention module is included, as well as the Jupyter Lab notebook used in the training, which includes modifications for generating confusion matrices.
        <li><b>For <a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Modificados/RetinaNet">RetinaNet 📂 </a></b></li> the following codes are included:
        <ul>
            <li><b>AumentoSIIM.ipynb 📄 :</b></li> Code to realize data augmentation for RetinaNet.
            <li><b>ConfuseMatrix.py 📄 :</b></li> Code developed for generating confusion matrices for RetinaNet.
            <li><b>trainMoidif.py 📄 :</b></li> odified code from train.py available on the <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet model's GitHub</a>, implementing patience usage, saving the model with the best performance, generating total loss and mAP graphs, and knowledge transfer.
            <li><b>modelModif.py 📄 : </b></li> Modified code from model.py available on the <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet model's GitHub</a>, implementing a draft of the unused attention module, but added in case it is wanted to explore and/or improve.
            <li><b>visualize_single_image.py 📄 :</b></li> Modified code from visualize_single_image.py available on the RetinaNet model's GitHub, which underwent corrections as it had faults when executed.
        </ul>
    </ul>  
</ul> 

<h2>Project Replication</h2>
<h3>In order to replicate this project it is recommended to follow the following steps:</h3>

First, we need to obtain the necessary image datasets for the project.

<ul>
    <li><b><a href="https://www.kaggle.com/datasets/nih-chest-xrays/data">Chest X-rays14</a></b></li> 
    <li><b><a href="https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data">RSNA Pneumonia</a></b></li>
    <li><b><a href="https://www.kaggle.com/c/siim-covid19-detection/data">SIIM-FISABIO-RSNA COVID19</a></b></li>
</ul>

To work, we need a virtual environment managed by Anaconda. For instructions on how to install Anaconda, please follow the official guide: <a href="https://docs.anaconda.com/anaconda/install/">Here</a>.

At this stage, we need three different environments to work on the project. For two of them, it is necessary to follow the required GitHub instructions to run the models:

<ul>
    <li><b><a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a></b></li> 
    <li><b><a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a></b></li>
</ul>

The last environment is required to replicate the exploratory analysis and manage the data. 
We need to download the env.yml file contained in this repository.

Then, to create the environment, run the following command in the Anaconda shell:

```bash
conda env create -f path/env.yml
```
Once we have the necessary environments, we can perform various tasks.

<h3>Replicate the exploratory analysis.</h3>

We need to work with the Anaconda environment previously created using the env.yml file and start JupyterLab by running the following command within the environment:

```bash
jupyterlab
```

With JupyterLab running, we have access to a data directory opened in the web browser. According to the exploratory analysis we want to replicate, we need two main things:

First, download the Jupyter Notebook related to the desired analysis, available in the project repository <a href="https://github.com/JairMathAI/COVID/tree/main/Exploratorios">here</a>. 

Second, for each analysis, the Notebook requires the relevant information for each dataset, which is available in the corresponding dataset folder in this repository <a href="https://github.com/JairMathAI/COVID/tree/main/Datos">here</a>.

To run the respective Notebook for each dataset, the following information file is needed:

<a href="https://github.com/JairMathAI/COVID/blob/main/Exploratorios/Exploratorio_RSNA.ipynb">Exploratorio_RSNA.ipynb</a> requires the path to the <a href="https://github.com/JairMathAI/COVID/blob/main/Datos/RSNA/Todo_info.csv">Todo_info.csv</a> file.<br><br>&nbsp;&nbsp;
<a href="https://github.com/JairMathAI/COVID/blob/main/Exploratorios/Exploratorio_Xray14.ipynb">Exploratorio_Xray14.ipynb</a> requires the path to the <a href="https://github.com/JairMathAI/COVID/blob/main/Datos/Xray14/Todas.csv">Todas.csv</a> and <a href="https://github.com/JairMathAI/COVID/blob/main/Datos/Xray14/bboxes.csv">bboxes.csv</a> files.<br>

<a href="https://github.com/JairMathAI/COVID/blob/main/Exploratorios/Exploratorio_SIIM.ipynb">Exploratorio_SIIM.ipynb</a> requires the path to the original train_image_level.csv and train_study_level.csv files provided with the origininal dataset.<br>

<b>Importan Note:</b><br>

For the correct execution of the notebooks, it is required that the paths to the directories containing the images are updated in the current Notebook and that the images are in the correct format.

<h3>Preparing images and annotations</h3>

To run the notebooks for the exploratory analysis and model training, we need to transform the images of each dataset appropriately, ensuring they meet the required format and size, and generate the provided annotation files, for example.


You can use the codes in the <a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Datos">Codigos_Datos</a> folder to adjust the images to the correct size and format. Alternatively, you can run your own script that resizes the images and saves them in PNG format.

The <a href="https://github.com/JairMathAI/COVID/tree/main/Anotaciones">Anotaciones:</a> folder contains the necessary annotations for the direct execution of <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a>, as explained in the respective GitHub project. For training with <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a>., a YAML file is required, as detailed in the corresponding GitHub documentation. The annotations for <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a>. can be derived from the provided <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a> annotations using the codes available in the <a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Datos">Codigos_Datos</a>  folder. Alternatively, we can start with the data summary information in the <a href="https://github.com/JairMathAI/COVID/tree/main/Datos">Datos:</a> folder and format the annotations as needed using a custom script, taking into account the next coments required for train <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a>:

The YAML file is the working datased indicator for <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a> and it need the folowing structure:

<b>path:</b> root/path/to/dataset/folder<br>
<b>train:</b> images/train<br>
<b>val:</b> images/validation<br>
<b>test:</b> images/test<br><br>

names:<br>

  0: Typical Appearance<br>
  1: Negative for Pneumonia<br>
  2: Indeterminate Appearance<br>
  3: Atypical Appearance<br><br>

In the other hand the structure that the dataset need to have and that we need to construct using the respective <a href="https://github.com/JairMathAI/COVID/tree/main/Anotaciones">Anotaciones</a> files is:

📂 Dataset  
├── 📂 images  
│   ├── 📂 train  
│   │   ├── 📄 tr_1.png  
│   │   ├── ...  
│   │   ├── 📄 tr_n.png  
│   ├── 📂 test  
│   │   ├── 📄 ts_1.png  
│   │   ├── ...  
│   │   ├── 📄 ts_n.png  
│   ├── 📂 validation  
│   │   ├── 📄 v_1.png  
│   │   ├── ...  
│   │   ├── 📄 v_n.png  
├── 📂 labels  
│   ├── 📂 train  
│   │   ├── 📄 tr_1.txt  
│   │   ├── ...  
│   │   ├── 📄 tr_n.txt  
│   ├── 📂 test  
│   │   ├── 📄 ts_1.txt  
│   │   ├── ...  
│   │   ├── 📄 ts_n.txt  
│   ├── 📂 validation  
│   │   ├── 📄 v_1.txt  
│   │   ├── ...  
│   │   ├── 📄 v_n.txt

Each row (one per boundingbox) in the .txt label file need to follow the next structure:<br>

class(int)&nbsp;&nbsp;  x_c(float)&nbsp;&nbsp;  y_c(float)&nbsp;&nbsp;   w_n(float) &nbsp;&nbsp; h_n(float)<br><br>

where:<br><br>
x_c: is the x coordinate of the center of the bounding box and it's normalized with respect to the the image<br>
y_c: is the y coordinate of the center of the bounding box and it's normalized with respect to the the image<br>
w_n: is the width of the bounding box normalized with respect to the width of the image<br>
h_n: is the height of the bounding box normalized with respect to the height of the image<br><br>

to obtain this format is important recall that the  <a href="https://github.com/JairMathAI/COVID/tree/main/Anotaciones">Anotaciones</a> files has the format needed for <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a>:<br><br>

Each row (one per boundingbox) in the .csv label file follow the next structure:<br>

image_path(str)&nbsp;&nbsp;  x_1(int)&nbsp;&nbsp;  y_1(int) &nbsp;&nbsp;  x_2(int) &nbsp;&nbsp; y_2(int)&nbsp;&nbsp;   class(string) <br><br>

where:<br><br>
image_path: is the path to the images corresponding to the annotation.<br>
x_1: is the x coordinate of the upper left corner of the bounding box<br>
y_2: is the y coordinate of the upper left corner of the bounding box<br>
x_2: is the x coordinate of the lower right corner of the bounding box<br>
y_2: is the y coordinate of the lower right corner of the bounding box<br><br>

Note that the <a href="https://github.com/JairMathAI/COVID/tree/main/Anotaciones">Anotaciones</a> files contain annotatin based in a image size of 640 x 640.

<h3>Training Models</h3>

With the corresponding images and annotations in order, we can follow the instructions in the respective model repository to train the model:

<ul>
    <li><b><a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a></b></li> 
    <li><b><a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a></b></li>
</ul>

<h3>Replication of results</h3>

In order to replicate the training results reported in the master thesis and given in the <a href="https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link">Google Drive folder</a>:<br>

<h4>For <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O:</a></h4>

We can utilize the <a href="https://github.com/JairMathAI/COVID/blob/main/Codigos_Modificados/YOLO/EntrenaYOLO.ipynb">EntrenaYOLO.ipynb</a> notebook to train the model and generate the corresponding confusion matrices. Please note that some paths need to be updated to run the generation code. In the respective training <a href="https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link">Google Drive folder</a>, a train.txt file is provided, which contains the specific hyperparameters used for each training session. Additionally, there is a .pth file corresponding to the generated model weights, allowing us to acces the models without training from scratch.


To replicate the results with the modified architecture, we need to follow the instructions provided in the <a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Modificados/YOLO">Codigos_Modificados</a> folder related to <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a>.

<h4>For <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a></h4>

We can use the respective annotation CSV file with the updated image paths provided in the <a href="https://github.com/JairMathAI/COVID/tree/main/Anotaciones">Anotaciones</a> folder. By following the instructions in the original <a href="https://github.com/yhenon/pytorch-retinanet">GitHub</a> repository, we can effectively implement the model. The <a href="https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link">Google Drive folder</a> also contains the weights for each trained model, and the corresponding hyperparameters used during training are detailed in the master's thesis (a link will be included once the work is officially published).

If we want to reproduce the results by applying data augmentation, modifying the training process, or visualizing and generating confusion matrices, we can follow the instructions in the <a href="https://github.com/JairMathAI/COVID/tree/main/Codigos_Modificados/RetinaNet">Codigos_Modificados</a> folder related to <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a>.

<h2>Important Note</h2>

Most of the code was initially developed to run local experiments as part of the final report. However, efforts are currently underway to enhance its reusability and reduce its dependency on local configurations.
