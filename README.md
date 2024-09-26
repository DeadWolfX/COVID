<h2>Project description</h2>
<h3>Here are the codes used in the development of the master's thesis titled "Deep Learning for the Identification and Localization of COVID-19 Induced Damage in Chest X-Rays." The folder structure in this main folder is as follows:</h3>
<ul>
    <li><b>Annotations:</b></li> Folder containing the various annotations used to train deep learning models. For each dataset, there is a folder containing annotations exclusively for RetinaNet, as the annotations for <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a> are generated from the files in the Data folder and the codes in the Codes_Data folder, both present in this repository. Each compressed file contains a text file with the description of the corresponding annotations.
    <li><b>Codes_Data:</b></li> Folder containing three Jupyter Lab notebooks, each with codes and functions used to generate the annotations contained in the Annotations folder and data processing for each set.
    <li><b>Data:</b></li> Contains text files for each dataset used, summarizing relevant information about each dataset in a structured manner.
    <li><b>Exploratory:</b></li> Contains three Jupyter Lab notebooks, one for each dataset used. These notebooks contain the codes and functions used to perform the reported exploratory analyses.
    <li><b>env.yml:</b></li> Anaconda virtual environment configuration containing all the necessary libraries for exploratory analysis, visualization, data preprocessing, and label manipulation.
    <li><b>Trainings:</b></li> The weights of the trained models described in the thesis, as well as the complete training information and graphics, are available in the following <a href="https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link">Google Drive folder</a>
    <li><b>Modified_Codes:</b></li>Contains two folders, one for each model:
    <ul>
        <li><b>For <a href="https://github.com/ultralytics/ultralytics">Y.O.L.O</a> </b></li> the code with modifications in the attention module is included, as well as the Jupyter Lab notebook used in the training, which includes modifications for generating confusion matrices.
        <li><b>For <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet</a></b></li> the following codes are included:
        <ul>
            <li><b>AumentoSIIM.ipynb:</b></li> Code to realize data augmentation for RetinaNet.
            <li><b>ConfuseMatrix.py:</b></li> Code developed for generating confusion matrices for RetinaNet.
            <li><b>trainMoidif.py:</b></li> odified code from train.py available on the <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet model's GitHub</a>, implementing patience usage, saving the model with the best performance, generating total loss and mAP graphs, and knowledge transfer.
            <li><b>modelModif.py: </b></li> Modified code from model.py available on the <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet model's GitHub</a>, implementing a draft of the unused attention module, but added in case it is wanted to explore and/or improve.
            <li><b>visualize_single_image.py:</b></li> Modified code from visualize_single_image.py available on the RetinaNet model's GitHub, which underwent corrections as it had faults when executed.
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

<a href="https://github.com/JairMathAI/COVID/blob/main/Exploratorios/Exploratorio_RSNA.ipynb">Exploratorio_RSNA.ipynb</a> requires the path to the <a href="https://github.com/JairMathAI/COVID/blob/main/Datos/RSNA/Todo_info.csv">Todo_info.csv</a> file.<br><br>
<a href="https://github.com/JairMathAI/COVID/blob/main/Exploratorios/Exploratorio_Xray14.ipynb">Exploratorio_Xray14.ipynb</a> requires the path to the <a href="https://github.com/JairMathAI/COVID/blob/main/Datos/Xray14/Todas.csv">Todas.csv</a> and <a href="https://github.com/JairMathAI/COVID/blob/main/Datos/Xray14/bboxes.csv">bboxes.csv</a> files.<br>

<a href="https://github.com/JairMathAI/COVID/blob/main/Exploratorios/Exploratorio_SIIM.ipynb">Exploratorio_SIIM.ipynb</a> requires the path to the original train_image_level.csv and train_study_level.csv files provided with the origininal dataset.<br>

<b>Importan Note:</b><br>

For the correct execution of the notebooks, it is required that the paths to the directories containing the images are updated in the current Notebook and that the images are in the correct format.
