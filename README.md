<H3>ENTER YOUR NAME: R0gith. K</H3>
<H3>ENTER YOUR REGISTER NO: 212223110042</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 28/08/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:
## DataSet:
![Screenshot 2024-09-04 113023](https://github.com/user-attachments/assets/57ad8f3a-556d-4bf5-b30d-83efa3d388c6)


## Dropping The Unwanted DataSet:
![Screenshot 2024-09-04 113050](https://github.com/user-attachments/assets/fe352783-c6d0-4ec3-98f8-89e013e2bca7)


## Checking Null Values:
![Screenshot 2024-09-04 113106](https://github.com/user-attachments/assets/76d84bd9-1ac0-4637-b3c5-162a280a24a2)


## Checking For Duplication:
![Screenshot 2024-09-04 113122](https://github.com/user-attachments/assets/0fb3a7ed-1e85-4515-8ca7-f0416fd82052)


## Describing The DataSet:
![Screenshot 2024-09-04 113141](https://github.com/user-attachments/assets/8e16722c-ab29-408a-96a8-45f9c9a91fc5)


## Scaling The DataSet:
![Screenshot 2024-09-04 113201](https://github.com/user-attachments/assets/dc688e2b-b7b5-4264-b5ea-56fbcf3c8274)


## X Features:
![Screenshot 2024-09-04 113214](https://github.com/user-attachments/assets/951bfbc5-748c-42b7-973d-180578902c82)


## Y Features:
![Screenshot 2024-09-04 113228](https://github.com/user-attachments/assets/3ddae6bb-30ce-4702-94f8-66f0858ab31f)


## Splitting The Training And Testing DataSet:
![Screenshot 2024-09-04 113248](https://github.com/user-attachments/assets/286e603b-52b4-4dd8-b701-34716003f761)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
