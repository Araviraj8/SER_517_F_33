# SER_517_F_33
**Capstone Project Spring 2024 - Group F-33**

**Members**:
Arnav Raviraj; araviraj@asu.edu
Adit Sandeep Virkar; avirkar@asu.edu,
Shivanjay Vilas Wagh; swagh5@asu.edu,
Vinay Kantilal Chavhan; vchavhan@asu.edu


**Topic**: Detecting intrusion in softwarized 5G networks using machine learning.

**Research Title** : 5G-NIDD: A Comprehensive Analysis of Network Intrusion Detection Algorithms on 5G Wireless Network Dataset

**Sponsor**: Abdallah Moubayed; abdallah.moubayed@asu.edu

Research Paper provided by Sponsor: 5G-NIDD: A comprehensive network Intrusion detection dataset generated over 5G Wireless network. 
Link: https://github.com/Araviraj8/SER_517_F_33/blob/main/related%20papers/Paper%20given%20by%20Sponsor.pdf

Research Paper completed by us: 5G-NIDD: A comprehensive network Intrusion detection dataset generated over 5G Wireless network.
https://github.com/Araviraj8/SER_517_F_33/blob/main/5G_NIDD__A_Comprehensive_Network_Intrusion_Detection_Dataset_Generated_over_5G_Wireless_Network.pdf

**Hardware Used**:
Processor: Intel Core i7
RAM: 8 GB DDR4
Storage: 512 GB SSD
Operating System: Windows 11

**Pre-requisites to run the application:**

1) You should have python3 installed or install Pycharm IDE with version 2022.2.1 (Community Edition) or the latest.
2) Conda should be installed with the latest version 4.3.16.
3) Download processed datasets from the link : https://drive.google.com/drive/u/4/folders/1lE9hLSn3J51t4eKuORW64M7J5bUPTDKG
   Make sure the combined.csv and processed_multiclass2.csv are in same directory i.e. Final Models folder
4) All the required model weights are present in the Model Weights Folder. 
**Install conda:**
  ```shell
pip install conda
 ```

**install dependencies using conda**:

  ```shell
conda env create -f environment.yml
 ```

**Binary Classification Models :**

**Go to Final Models folder:**

```shell
cd Final Models
 ```

**For XgBoost:**
 ```shell
 python xgb_final.py
 ```
 
**For CatBoost:**
```shell
python CatBoostFinal.py
```

**For ADABoost:**
```shell
python Adaboost_final.py
```

**For ANN:**
```shell
python ann_binary_final.py
```

**For GAN:**
```shell
python GAN_Binary.py
```

