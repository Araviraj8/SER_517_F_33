# SER_517_F_33
Capstone Project Spring 2024 - Group F-33

Members:
Arnav Raviraj; araviraj@asu.edu
Adit Sandeep Virkar; avirkar@asu.edu,
Shivanjay Vilas Wagh; swagh5@asu.edu,
Vinay Kantilal Chavhan; vchavhan@asu.edu


Topic: Detecting intrusion in softwarized 5G networks using machine learning.

Sponsor: Abdallah Moubayed; abdallah.moubayed@asu.edu

Research Paper provided by Sponsor: 5D-NIDD: A comprehensive network Intrusion detection dataset generated over 5G Wireless network. 
Link: https://github.com/Araviraj8/SER_517_F_33/blob/main/related%20papers/Paper%20given%20by%20Sponsor.pdf

Hardware Used:
Processor: Intel Core i7
RAM: 8 GB DDR4
Storage: 512 GB SSD
Operating System: Windows 11

**install dependencies**:

  ```shell
conda env create -f environment.yml
 ```

**Binary Classification**

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
**For GAN Model with CatBoost Classifier:**

```shell
python GAN_Binary_CatBoost.py
```

