Aquí se encuentran los códigos utilizados en el desarrollo de la tesis de maestría titulada "Aprendizaje profundo para la identificación y localización de daños por COVID19 en radiografías de rayos X". La estructura de carpetas en esta carpeta principal es la siguiente:

	Anotaciones:
    
    	Carpeta que contiene las distintas anotaciones utilizadas para entrenar los modelos de aprendizaje profundo, para cada conjunto de datos existe una carpeta que contiene las anotaciones para RetinaNet ya que para YOLO es sencillo generar las anotaciones usando los archivos de la carpera Datos y los códigos contenidos en la carpeta Codigos_Datos ambos presentes en este repositorio, cada commprimido contiene un txt con la descripcion del contenido.
    
    Codigos_Datos:
    
    	Carpeta que contiene tres cuadernos de Jupyter Lab, cada uno con códigos y funciones tilizadas para generar las anotaciones contenidas en la carpeta anteriormente descrita.
    
    Datos:
    
    	Contiene archivos .txt respectivos a cada conjunto de datos utilizados, donde se resume de manera estructurada información relevante sobre cada conjunto de datos utilizado.
    
    Exploratorios:
    
    	Contiene tres cuadernos de Jupyter Lab, uno por cada conjunto de datos utilizado. Estos cuadernos contienen los códigos y funciones utilizados para realizar los respectivos análisis exploratorios.

    env.yml:
    
        Configuración de ambiente virtual de anaconda que contiene todas la librerías necesarias para los analisis exploratorios, visualización, preprocesamiento de datos y manipulacion de etiquetados.

    Entrenamientos:

        Los pesos de los modelos entrenados descritos en la tesis así como las gráficas e informacion de entrenamineto completo esta disponible en la crpeta de drive:
            https://drive.google.com/drive/folders/1JVR-FKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link
    
    Codigos_Modificados:

        Contine dos carpetas una para cada modelo donde para el caso de YOLO se incluye el codigo con las capas de convolución agregadas para probar el modelo y tambien el cuaderno de Jupyter Lab con la que se realizo el entrenamiento y las matrices de confusión.

        Para RetinaNet se incluyen los códigos:
             ConfuseMatrix.py que se genro para la cración de matrices de confusion que originalmente no se incluye en el GuitHub del modelo.
             trainMoidif.py  codigo modificado de train.py disponible en el GuitHub del modelos que implemnta el uso de patience, el guardado del modelo con mejor desempeño, la generacin de la grafica de perdida total y el mAP promedio.
             modelModif.py codigo modificado de model.py disponible en el GuitHub del modelos qe implementa un bosqueo del modulo de atencion no probado ya que se decidio no usar pero se añade dado el caso que se quiera  explorar y/o mejorar.



    EntrenaYOLO.ipynb:

        Libreta de Jupyter Lab que contiene el codigo para entrenar el modelo de YOLO, hacer visualizaciones y la generación de matrices de confusión acorde a cada etiquetado.
    
   



