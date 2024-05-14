Aquí se encuentran los códigos utilizados en el desarrollo de la tesis de maestría titulada "Estrategias de aprendizaje profundo para la identificación y localización de daños generados por COVID19 usando radiografías de rayos X."
La estructura de carpetas en esta carpeta principal es la siguiente:

Anotaciones: 
    
    Carpeta que contiene las distintas anotaciones utilizadas para entrenar los modelos de aprendizaje profundo, para cada conjunto de datos existe una carpeta que contiene las anotaciones para RetinaNet ya que para YOLO es sencillo generar las anotaciones usando los archivos de la carpera Datos y los códigos contenidos en la carpeta Códigos_Datos ambos presentes en este repositorio, cada commprimido contiene un archivo de texto con la descripcion del contenido.

Codigos_Datos: 
    
    Carpeta que contiene tres cuadernos de Jupyter Lab, cada uno con códigos y funciones tilizadas para generar las anotaciones contenidas en la carpeta anteriormente descrita y el procesamiento de datos para cada conjunto.

Datos: 

    Contiene archivos de texto respectivos a cada conjunto de datos utilizados, donde se resume de manera estructurada la información relevante sobre cada conjunto de datos.

Exploratorios: 

    Contiene tres cuadernos de Jupyter Lab, uno por cada conjunto de datos utilizado. Estos cuadernos contienen los códigos y funciones utilizados para realizar los respectivos análisis exploratorios.

env.yml: 

    Configuración de ambiente virtual de anaconda que contiene todas la librerías necesarias para los analisis exploratorios, visualización, preprocesamiento de datos y manipulacion de etiquetados.

Entrenamientos: 

    Los pesos de los modelos entrenados descritos en la tesis así como las gráficas e informacion de entrenamineto completo esta disponible en la crpeta de drive que se enceuntra en el siguiente enlace 
    https://drive.google.com/drive/foldFKDxJcaKLuTDaTM2A9S_6f6m2J4A?usp=drive_link

Códigos_Modificados: 

    Contiene dos carpetas, una para cada modelo. Para el caso de YOLO, se incluye el código con las capas de convolución agregadas para probar el modelo, así como el cuaderno de Jupyter Lab utilizado en el entrenamiento y para la genetración de matrices de confusión.
    Para RetinaNet, se incluyen los códigos: ConfuseMatrix.py, generado para la creación de matrices de confusión, originalmente no incluido en el GitHub del modelo.
    
    trainMoidif.py, código modificado de train.py disponible en el GitHub del modelo, que implementa el uso de paciencia, el guardado del modelo con mejor desempeño, la generación de la gráfica de pérdida total y el mAP promedio, así como la tranferencia de conocimiento.

    modelModif.py, código modificado de model.py disponible en el GitHub del modelo, que implementa un bosquejo del módulo de atención no utilizado, pero se añade por si se quisiera explorar y/o mejorar.
