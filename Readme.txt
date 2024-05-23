###########################################################################################English#######################################################################################################################################

Here are the codes used in the development of the master's thesis titled "Deep Learning Strategies for the Identification and Localization of COVID19-Induced Damage Using X-ray Radiographs." The folder structure in this main folder is as follows:

Annotations:

    Folder containing the various annotations used to train deep learning models. For each dataset, there is a folder containing annotations exclusively for RetinaNet, as the annotations for Y.O.L.O are generated from the files in the Data folder and       the codes in the Codes_Data folder, both present in this repository. Each compressed file contains a text file with the description of the corresponding annotations.

Codes_Data:

    Folder containing three Jupyter Lab notebooks, each with codes and functions used to generate the annotations contained in the Annotations folder and data processing for each set.

Data:

    Contains text files for each dataset used, summarizing relevant information about each dataset in a structured manner.

Exploratory:

    Contains three Jupyter Lab notebooks, one for each dataset used. These notebooks contain the codes and functions used to perform the reported exploratory analyses.

env.yml:

    Anaconda virtual environment configuration containing all the necessary libraries for exploratory analysis, visualization, data preprocessing, and label manipulation.

Trainings:

    The weights of the trained models described in the thesis, as well as the complete training information and graphics, are available in the Google Drive folder at the following link: 
    https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link

Modified_Codes: 
    Contains two folders, one for each model:
    
    For Y.O.L.O, the code with modifications in the attention module is included, as well as the Jupyter Lab notebook used in the training, which includes modifications for generating confusion matrices.

    For RetinaNet, the following codes are included:

        ConfuseMatrix.py:

            Code developed for generating confusion matrices for RetinaNet.

        trainMoidif.py: 

            Modified code from train.py available on the RetiaNet model's GitHub, implementing patience usage, saving the model with the best performance, generating total loss and mAP graphs, and knowledge transfer.
    
        modelModif.py: 

            Modified code from model.py available on the RetinaNet model's GitHub, implementing a draft of the unused attention module, but added in case it is wanted to explore and/or improve.

        visualize_single_image.py: 

            Modified code from visualize_single_image.py available on the RetinaNet model's GitHub, which underwent corrections as it had faults when executed.

        Note.csv: 

            File containing information about the execution of ConfuseMatrix.py.

########################################################################################Spanish#############################################################################################################################################

Aquí se encuentran los códigos utilizados en el desarrollo de la tesis de maestría titulada "Estrategias de aprendizaje profundo para la identificación y localización de daños generados por COVID19 usando radiografías de rayos X."
La estructura de carpetas en esta carpeta principal es la siguiente:

Anotaciones: 

    Carpeta que contiene las distintas anotaciones utilizadas para entrenar los modelos de aprendizaje profundo, para cada conjunto de datos existe una carpeta que contiene las anotaciones únicamente para RetinaNet ya que las anotaciones para Y.O.L.O       se generan a partir de los archivos contenidos en la carpera Datos y los códigos contenidos en la carpeta Códigos_Datos ambos presentes en este repositorio, cada commprimido contiene un archivo de texto con la descripcion de las anotaciones             correspondientes.

Codigos_Datos: 

    Carpeta que contiene tres cuadernos de Jupyter Lab, cada uno con códigos y funciones utilizadas para generar las anotaciones contenidas en la carpeta Anotaciones y el procesamiento de datos para cada conjunto.

Datos: 
    
    Contiene archivos de texto respectivos a cada conjunto de los conjuntos datos utilizados, donde se resume de manera estructurada la información relevante sobre cada conjunto.

Exploratorios: 

    Contiene tres cuadernos de Jupyter Lab, uno por cada conjunto de datos utilizado. Estos cuadernos contienen los códigos y funciones utilizados para realizar los análisis exploratorios reportados.

env.yml: 

    Configuración de ambiente virtual de anaconda que contiene todas la librerías necesarias para los analisis exploratorios, visualización, preprocesamiento de datos y manipulacion de etiquetados.

Entrenamientos: 

    Los pesos de los modelos entrenados descritos en la tesis así como las gráficas e informacion de entrenamineto completa esta disponible en la carpeta de drive que se enceuntra en el siguiente enlace: 

    https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link

Códigos_Modificados: Contiene dos carpetas, una para cada modelo:

    Para el caso de Y.O.L.O, se incluye el código con las modificaciones en el módulo de atención, así como el cuaderno de Jupyter Lab utilizado en el entrenamiento que contempla las modificaciones para la generación de matrices de confusión.

    Para RetinaNet, se incluyen los códigos:

        ConfuseMatrix.py: 
            
            Código desarrollado para la generacion de matrices de cofusión para RetinaNet.

        trainMoidif.py: 

            Código modificado de train.py disponible en el GitHub del modelo RetiaNet, que implementa el uso de paciencia, guardado del modelo con mejor desempeño, la generación de gráficas de pérdida total y mAP, así como la tranferencia de                        conocimiento.

        modelModif.py: 
            
            Código modificado de model.py disponible en el GitHub del modelo RetinaNet, que implementa un bosquejo del módulo de atención no utilizado, pero se añade por si se quisiera explorar y/o mejorar.

        visualize_single_image.py: 

            Código modificado de visualize_single_image.py disponible en el GitHub del modelo RetinaNet, al cual se le realizo correcciones ya que al ejecutarse este presenta fallos.

        Nota.csv: 

            Archivo que contiene infromación sobre la ejecusión de ConfuseMatrix.py.
