<h2>Note</h2>

<p>The train code presented here should replace the train code found in the installation of the original model repository. This code includes new implementations that can be utilized using the following brief description, as it is documented in the code, with further details provided in the original model repository:</p>

python train.py --dataset csv --csv_train path_to_training_csv --csv_classes path_to_csv_defining_classes --csv_val path_to_validation_csv --epochs number_of_epochs (integer) --depth resnet_depth --num_save how_often_to_save_model (if set to 0, only saves the model with the highest map) --optimizer optimizer (can be SGD, ADAMW, or ADAM) --patience (integer indicating the number of epochs the model should train without improvement; if these epochs are exceeded, training is stopped and the model is saved)


<p>A code for generating confusion matrices is implemented, which should simply be placed inside the repository folder and used with:</p>


python ConfuseMatrix.py --test_csv path_to_test_csv --model_path path_to_model_file_.pt --class_list path_to_csv_defining_classes --mod_pred model_prediction_mode --save_path path_to_save_matrices --mode specifies_type_of_set_used_to_generate_matrices (can be: test or validation)






