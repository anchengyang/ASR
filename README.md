# TIL2023 ASR Qualifiers Challenge

The TIL2023 ASR Qualifiers code repository

## Setting up the environment

### Clone Repository   
Run this command to clone the repository  
   
```shell
git clone https://github.com/til-23/til-23-asr.git
cd til-23-asr/
```    

<!--
### Creating a python environment     
Proceed to create a python virtual environment. Ensure that you are inside in repository
     
```shell
cd til-asr/
pip install virtualenv
virtualenv venv
```

### Activate the python environment   
To activate the python environment, run this command:   
     
```shell
source venv/bin/activate
```
-->

### Install requirements    
To install the requirements, run the following commands:    
    
```shell
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116   
``` 

## Running the code

### Model Training   
To train a model, modify the directory where you saved the data and the corresponding annotation file,  modify the model path directory where the trained model is saved, experiment with the different `hparams` values. Feel free to modify the code too. Execute this command to start training:   

```shell
python3 src/train.py
```
   
### Model Inference  
After training the model, modify the path of the test data and the model. You can execute this command to do inference:   
     
```shell
python3 src/inference.py
```  
   
