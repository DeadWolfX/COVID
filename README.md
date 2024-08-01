<h2>Project description</h2>
<h3>Here are the codes used in the development of the master's thesis titled "Deep Learning for the Identification and Localization of COVID-19 Induced Damage in Chest X-Rays." The folder structure in this main folder is as follows:</h3>
<ul>
    <li><b>Annotations:</b></li> Folder containing the various annotations used to train deep learning models. For each dataset, there is a folder containing annotations exclusively for RetinaNet, as the annotations for Y.O.L.O are generated from the files in the Data folder and the codes in the Codes_Data folder, both present in this repository. Each compressed file contains a text file with the description of the corresponding annotations.
    <li><b>Codes_Data:</b></li> Folder containing three Jupyter Lab notebooks, each with codes and functions used to generate the annotations contained in the Annotations folder and data processing for each set.
    <li><b>Data:</b></li> Contains text files for each dataset used, summarizing relevant information about each dataset in a structured manner.
    <li><b>Exploratory:</b></li> Contains three Jupyter Lab notebooks, one for each dataset used. These notebooks contain the codes and functions used to perform the reported exploratory analyses.
    <li><b>env.yml:</b></li> Anaconda virtual environment configuration containing all the necessary libraries for exploratory analysis, visualization, data preprocessing, and label manipulation.
    <li><b>Trainings:</b></li> The weights of the trained models described in the thesis, as well as the complete training information and graphics, are available in the following <a href="https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link">Google Drive folder</a>
    <li><b>Modified_Codes:</b></li>Contains two folders, one for each model:
    <ul>
        <li><b>For Y.O.L.O</b></li> the code with modifications in the attention module is included, as well as the Jupyter Lab notebook used in the training, which includes modifications for generating confusion matrices.
        <li><b>For RetinaNet</b></li> the following codes are included:
        <ul>
            <li><b>AumentoSIIM.ipynb:</b></li> Code to realize data augmentation for RetinaNet.
            <li><b>ConfuseMatrix.py:</b></li> Code developed for generating confusion matrices for RetinaNet.
            <li><b>trainMoidif.py:</b></li> odified code from train.py available on the RetiaNet model's GitHub, implementing patience usage, saving the model with the best performance, generating total loss and mAP graphs, and knowledge transfer.
            <li><b>modelModif.py: </b></li> Modified code from model.py available on the <a href="https://github.com/yhenon/pytorch-retinanet">RetinaNet model's GitHub</a>, implementing a draft of the unused attention module, but added in case it is wanted to explore and/or improve.
            <li><b>visualize_single_image.py:</b></li> Modified code from visualize_single_image.py available on the RetinaNet model's GitHub, which underwent corrections as it had faults when executed.
        </ul>
    </ul>
 <li><b>Note.csv:</b></li>  File containing information about the execution of ConfuseMatrix.py.   
</ul>  
