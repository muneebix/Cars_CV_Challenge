# Cars_CV_Challenge
Dataset path= https://ai.stanford.edu/~jkrause/cars/car_dataset.html


Run Pipeline.py -m test/train -d 'diretcory for images to predict 'with arg mode train/test

Data Preprocessing:

    preprocess.py ==> Generate train_/test_ folders for training/testing the model through annotation files (.mat files)
    provided with the cars dataset
    

Test and Validation Split:

split_test_val.py ==>    Split test data into test and validation ,i used seperate validation images for validation purpose.


    
    
Requirements 

 ==> tensorflow
 
 ==> keras
 
 ==> pandas
 
 ==> opencv
 
 ==> Numpy
