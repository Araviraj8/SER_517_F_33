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

To replicate all the results run all the files in the Final Models folder.
For example, for running XGBoost for the top 25 features for binary classification, follow the below steps:
1) Install the necessary Python packages: pandas, scikit-learn (sklearn), matplotlib, numpy, xgboost, seaborn
2) Clone the repository to your local machine using Git.
3) Open a terminal or command prompt and navigate to the directory containing the cloned repository.
4) Place the dataset file (Combined.csv) in the same directory as the code.
5) Run the Python script (xgb_final.py) in your preferred Python IDE or text editor.
6) After running the script, you will see the results of the XGBoost model printed on the console.

For running AdaBoost for the top 25 features for binary classification, follow the below steps:
1) To run the code in the Jupyter Notebook, you need the following libraries: pandas, scikit-learn.
2) Clone the repository to your local machine using Git.
3) Open a terminal or command prompt and navigate to the directory containing the cloned repository.
4) Place the dataset file in the same directory as the code.
5) Open and run the `ADAboost final.ipynb` notebook using Jupyter Notebooks.
6) After running the script, you will see the results of the AdaBoost model printed on the console.

Similarly, for running ANN using Tensorflow for the Pearson correlation for binary classification, follow the below steps:
1) To run the code in the Jupyter Notebook, you need the following libraries: pandas, scikit-learn, tensorflow
2) Clone the repository to your local machine using Git.
3) Open a terminal or command prompt and navigate to the directory containing the cloned repository.
4) Place the dataset file in the same directory as the code.
5) Open and run the `ANN.ipynb` notebook using Jupyter Notebooks.
6) After running the script, you will see the results of the ANN(tensorflow) model printed on the console.

For running GAN for the top 25 featues for binary classification, follow the below steps:
1) Install the necessary Python packages: pandas, numpy, scikit-learn, catboost, tensorflow.
2) Clone the repository to your local machine using Git.
3) Open a terminal or command prompt and navigate to the directory containing the cloned repository.
4) Place the dataset file in the same directory as the code.
5) Run the Python script (GANBasedModel.py) in your preferred Python IDE or text editor.
6) After running the script, you will see the results of the GAN model printed on the console.

Once again, for running XGBoost for multiclass classification, follow the below steps:
1) Install the necessary Python packages: pandas, scikit-learn (sklearn), matplotlib, numpy, xgboost.
2) Clone the repository to your local machine using Git.
3) Open a terminal or command prompt and navigate to the directory containing the cloned repository.
4) Place the dataset file in the same directory as the code.
5) Run the Python script (xgb_final2.py) in your preferred Python IDE or text editor.
6) After running the script, you will see the results of the XGBoost model printed on the console.

For running ANN using PyTorch for multiclass classification, follow the below steps:
1) Install the necessary Python packages: pandas, scikit-learn, pytorch.
2) Clone the repository to your local machine using Git.
3) Open a terminal or command prompt and navigate to the directory containing the cloned repository.
4) Place the dataset file in the same directory as the code.
5) Run the Python script (train_ann.py) in your preferred Python IDE or text editor.
6) After running the script, you will see the results of the ANN model printed on the console.

Similarly you will be able to run the remaining files to obtain the results following the above steps. Also you can install dependencies like below:

install dependencies:

  ```shell
conda env create -f environment.yml
 ```

**Go to Final Models folder:**

```shell
cd Final Models
 ```

**For XgBoost:**

 ```shell
 python xgb_final.py
 ```
 

```shell
python CatBoostFinal.py
```


**GAN Model with CatBoost Classifier:**

```shell
python GAN_Binary_CatBoost.py
```

