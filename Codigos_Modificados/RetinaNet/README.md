<h2>Note</h2>

<p>The train code presented here should replace the train code found in the installation of the original model repository. This code includes new implementations that can be utilized using the following brief description, as it is documented in the code, with further details provided in the original model repository:</p>


```bash
python train.py --dataset csv --csv_train path_to_training_csv --csv_classes path_to_csv_defining_classes --csv_val path_to_validation_csv --epochs epochs --depth depth --num_save n --optimizer optimizer  --patience patience
```

where

  <ul>
  <li><b>epochs</b>(int):</li> the number of epochs
  <li><b>depth</b>(int):</li> the Retinanet depth (18, 34, 50, 101 or 152)
  <li><b>num_save</b>(int):</li> how often to save model (if set to 0, only saves the model with the highest map)
  <li><b>optimizer</b>(string):</li> optimizer used to train (SGD, ADAMW, or ADAM)
  <li><b>patience</b>(int):</li> the number of epochs the model should train without improvement; if these epochs are exceeded, training is stopped and the model is saved
  </ul>


<p>A code for generating confusion matrices is implemented, which should simply be placed inside the repository folder and used with:</p>


```bash
python ConfuseMatrix.py --test_csv path_to_test_csv --model_path path_to_model_file_.pt --class_list path_to_csv_defining_classes --mod_pred mood --save_path path_to_save_matrices --mode mode
```

where

  <ul>
  <li><b>mod_pred</b>(string):</li> kind of image labeling it can be: single (opacity images with boundingbox and no opacity images without annotation), total(opacity images has entire image anotation), Xray14 (ChestXray14 dataset), SIIM (Covid dataset)
  <li><b>mode</b>(string):</li> dataset type used in the matrix generatios, it can be test or validation
  </ul>





