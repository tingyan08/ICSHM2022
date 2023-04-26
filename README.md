# IC-SHM 2022
This repository is the implementation for the project 3 of International Competition for Structural Health Monitoring (IC-SHM). 
* The project website: https://shmc.tongji.edu.cn/ICSHM2022/main.htm.

## Project 3 Details

This project contains two tasks. Build a data-driven model to reconstruct the lost data, and evaluate the accuracy of the model.

### Task a: Data-driven modeling
1. Four sensors are normally operating all the time while one sensor is assumed broken for the whole time. The lost data should be reconstructed from the data measurement of the other normally operating sensors by building a data-driven model. Each participant should use the trained model to reconstruct the lost data of the broken sensor.

2. Time history data of five sensors are provided. The data of three sensors are assumed lost for the whole time. Each participant needs to reconstruct the lost data.


### Task b: Damage identification

1. Labelled data, that includes the bridge response under varied damage conditions, are provided as the training dataset. Six unlabeled datasets will also be provided as the testing dataset. Each participant needs to use the trained model to identify the damage degrees of the bridge for each testing data.


## Dependencies and Installation
* Clone thie repo:
```
git clone https://github.com/tingyan08/ICSHM2022.git
cd ICSHM2022
```

* Install the required dependencies:
```
pip install -r requirement.txt

```
## Feature Extraction
In this project, we tried to extract the infromative features using autoencoder with different custom losses. To train the model:
```
python3 train_acceleration_extraction.py --arch extraction --trainer <AE/DamageAE/TripletAE> --max_epoch <epoch> --description <some description>
```

```
python3 train_acceleration_extraction.py --arch extraction --trainer AE --max_epoch <epoch> --description <some description>
```


## Data-driven Modeling

In this study, we attempt to develop a reconstruction network by both displacement and acceleration signals. To train the models:

```
python3 train_displacement_reconstruction.py --arch reconsturction --trainer EncoderDecoder --load_model <None/AE/TripletAE/DamageAE> --transfer <True/False> --max_epoch <epoch> --description <some description>
```

```
python3 train_acceleration_reconstruction.py --arch reconsturction --trainer EncoderDecoder --load_model <None/AE> --transfer <True/False> --max_epoch <epoch> --description <some description>
```

## Damage Identification

In this project, only the displacement signals contains the information of structural damage. We have tried the classification and regresion task, the `arch` argument can be modified `regression/classification` to change the tasks. To train the models:

```
python3 train_displacement_reconstruction.py --arch <regression/classification> --trainer CNN --load_model <None/AE/TripletAE/DamageAE> --transfer <True/False> --max_epoch <epoch> --description <some description>
```

## Signal Generation
In this project, the testing displacement signals may from the unseen damage conditions compared to the training dataset. So, we adpoted a __WCGAN-GP__ to generate the signals under unseen damage scenarios. Then fine-tune the networks for damage identification. To train the model:

```
python3 train_displacement_generation.py --arch generation --trainer WCGAN_GP --max_epoch <epoch> --description <some description>
```

## Test

The codes for testing and outputing all the figures are from belows:

1. Feature Extraction: `testing_extraction.ipynb`
2. Signal Generation: `testing_generation.ipynb`
3. Signal Reconstruction: `testing_reconstruction.ipynb`
4. Damage Identification: `testing_quantification.ipynb`


