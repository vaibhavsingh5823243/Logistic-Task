import logging
import os
import pickle
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import  variance_inflation_factor
from logginghandler import Log
log_obj=Log("Main").log()

class HandlingMultipleFiles:

    def __init__(self, dirname=None):
        """"
        dirname:name of folder where all files are stored
        """
        self.dirname = dirname

    def skip_redundant_rows_nd_add_label(self, file_path, label):
        """
        This function will take two parameters
        :param file_path: path of data
        :param label: lable name to be assign to each instance
        :return: return clean data with lable
        """
        try:
            data = pd.read_csv(file_path, skiprows=4, error_bad_lines=False)
            if len(data.iloc[0])==7:
                 data['Lable'] = label
                 data.to_csv(file_path)
                 return data
            else:
                return True

        except Exception as e:
            log_obj.error(e)

    def mergefile(self):
        """
        This function will merge all the csv file of each folder
        and return a single file with lables.
        :return:
        """
        try:
            combinefile=pd.DataFrame()
            path=f"{self.dirname}\\combinefile.csv"
            for folder in os.listdir(self.dirname):
                folder_path = os.path.join(self.dirname, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        filepath = os.path.join(folder_path, file)
                        data = self.skip_redundant_rows_nd_add_label(file_path=filepath, label=folder)
                        if isinstance(data,bool):
                            print('Inside isinstance')
                            return pd.read_csv(path)
                        combinefile = pd.concat([combinefile, data])
            combinefile.to_csv(path)
            return combinefile

        except Exception as e:
            log_obj.error(e)

    def result(self):
        """
        This is function which will check if the file is already merged or not
        if True return already merged file
        else first call the mergefile function then return the merged file
        :return:
        """
        try:
            data=self
            return data

        except Exception as e:
            log_obj.error(e)


class DataPreprocessing:
    def __init__(self, data=None):
        self.data = data

    def handling_null_values(self):
        """
        This function will fill all the null values of
        each columns with their columns mean value
        :return:
        """
        try:
            null_values = self.data.isnull().sum() > 0
            columns = null_values[null_values == True].index
            for col in columns:
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col] = self.data[col].fillna(self.data[col].mean())

        except Exception as e:
            log_obj.error(e)

    def outlier_removal(self):
        """
        This function will remove all the outlier present in data
        :return:
        """
        try:
            def outlier_limits(col):
                try:
                    Q3, Q1 = np.nanpercentile(col, [75, 25])
                    IQR = Q3 - Q1
                    UL = Q3 + 1.5 * IQR
                    LL = Q1 - 1.5 * IQR
                    return UL, LL
                except Exception as e:
                    return e

            for column in self.data.columns:
                if self.data[column].dtype == 'int64':
                    UL, LL = outlier_limits(self.data[column])
                    self.data[column] = np.where((self.data[column] > UL) | (self.data[column] < LL), np.nan,self.data[column])


        except Exception as e:
            log_obj.error(e)

    def profile_report(self):
        """
        This function will create profile report of the data
        :return:
        """
        try:
            report = ProfileReport(self.data)
            report.to_file('templates\\profile_report.html')
            return report
        except Exception as e:
            log_obj.error(e)

    def vif(self):
        """
        This function will find vif of all the columns
        and return dataframe with features and their vif score
        :param data:
        :return:
        """
        try:
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(self.data.values, i) for i in range(self.data.shape[1])]
            vif['Columns'] = self.data.columns
            vif = sorted(vif.values, key=lambda x: (x[0], x[1]))
            vif = pd.DataFrame(vif, columns=['VIF', 'Features'])
            return vif

        except Exception as e:
            log_obj.error(e)

    def standardization(self):
        """
        This function will change the data into standard form
        :param data:
        :return:
        """
        try:
            features,target=self.split_feature_target()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(features)
            with open('scaler.pickle','wb') as fp:
                pickle.dump(scaler,fp)
            scaled_data = pd.DataFrame(scaled_data, columns=features.columns)
            return scaled_data
        except Exception as e:
            log_obj.error(e)

    def change_target(self,val,num_to_cat=False):
        """
        This function will change target values form categorical value
        to nominal value of vice versa.
        :param val: val to be changed
        :param num_to_cat: by default False if True then change the target
                           from numeric to categorical value
        :return: return the changed value
        """
        try:
            values=['bending1','bending2','cycling','lying','sitting','standing','walking']
            if num_to_cat:
                return values[val]
            else:
                for i in range(len(values)):
                    if values[i]==val:
                        return i
        except Exception as e:
            log_obj.error(e)

    def split_feature_target(self):
        """This function will split the dataset
        int feature and target part"""
        try:
            target=self.data['Lable']
            features=self.data.drop(columns=['Lable','# Columns: time','Unnamed: 0'])
            return features,target
        except Exception as e:
            log_obj.error(e)

    def split_into_train_test(self):
        """
        This function will split the features and target value into
        train and test part.
        :return: x_train,y_train,x_test,y_test
        """
        try:
            features,target=self.split_feature_target()
            features=self.standardization()
            x_train,x_test, y_train, y_test = train_test_split(features, target, test_size=.25, random_state=58)
            return x_train, y_train, x_test, y_test
        except Exception as e:
            log_obj.error(e)

    def outlier_detection(self):

        """This function will detect the outliers
        by ploting there boxplot"""

        try:
            fig, ax = plt.subplots(figsize=(20, 20))
            sns.boxplot(data=self.data, ax=ax)
            plt.title('Boxplot For Outlier Detection')
            plt.show()
        except Exception as e:
            log_obj.error(e)

    def operation(self):
        try:
            self.handling_null_values()
            self.outlier_removal()
            self.data['Lable']=self.data['Lable'].apply(self.change_target)
            x_train,y_train,x_test,y_test=self.split_into_train_test()
            return x_train,y_train,x_test,y_test
        except Exception as e:
            log_obj.error(e)


class Models:



    def model1(self,x_train,y_train):

        """This is the model with best accuracy"""

        try:
            mod1=LogisticRegression()
            mod1.fit(x_train,y_train)
            with open("model1.pickle","wb") as fp:
                 pickle.dump(mod1,fp)
            return True
        except Exception as e:
            log_obj.error(e)

    def model2(self,x_train,y_train,solver='lbfgs',max_iter=1000):

        """This is an alternate model you can
         configure it according to your needs"""

        try:
            mod2=LogisticRegression(solver=solver,max_iter=max_iter)
            mod2.fit(self.x_train,self.y_train)
            with open("model2.pickle","wb") as fp:
                pickle.dump(mod2,fp)
            return True
        except Exception as e:
            log_obj.error(e)

    def predict(self,feature,model=1):

        try:
            with open(f"model{model}.pickle","rb") as f:
                model=pickle.load(f)
                result=model.predict(feature)

                return result
        except Exception as e:
            log_obj.error(e)
