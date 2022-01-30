#        libraries
# ==================== #
import pandas as pd
import numpy as np
from scipy import stats
import time
import random
import math
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from lightgbm import LGBMClassifier
from python.KMeansFeaturer import KMeansFeaturizer

import warnings
warnings.filterwarnings("ignore")

class pipeline:

    def __init__(self, df, test):
        self.df = df
        self.test = test

    def probabilities(self,n):
        df= self.df
        test = self.test

        column = [f"geo_level_{n}_id"]
        nom1 = [f"prob1_geo{n}"]
        nom2 = [f"prob2_geo{n}"]
        nom3 = [f"prob3_geo{n}"]
        #This will save the probabilities in one column for each in df and dfOut
        damage1 = dict()
        damage2 = dict()
        damage3 = dict()

        for i, j in df[column].value_counts().iteritems():
            n1 = len(df[df.damage_grade == 1][df[column[0]] == i])
            n2 = len(df[df.damage_grade == 2][df[column[0]] == i])
            n3 = len(df[df.damage_grade == 3][df[column[0]] == i])

            damage1[i[0]] = n1/j
            damage2[i[0]] = n2/j
            damage3[i[0]] = n3/j

        list1 = []
        list2 = []
        list3 = []

        for i in df[column[0]]:
            list1.append(damage1.get(i))
            list2.append(damage2.get(i))
            list3.append(damage3.get(i))

        self.df[nom1[0]] = list1
        self.df[nom2[0]] = list2
        self.df[nom3[0]] = list3

        list1 = []
        list2 = []
        list3 = []

        for i in test[column[0]]:
            list1.append(damage1.get(i))
            list2.append(damage2.get(i))
            list3.append(damage3.get(i))

        self.test[nom1[0]] = list1
        self.test[nom2[0]] = list2
        self.test[nom3[0]] = list3

    def fit(self):
        target = 'damage_grade' 
        numeric = ['age_pt','area_percentage_pt', 'height_percentage_pt',
                    'prob1_geo1','prob1_geo2','prob1_geo3',
                    'prob2_geo1','prob2_geo2','prob2_geo3',
                    'prob3_geo1','prob3_geo2','prob3_geo3', 
                    'CntFloorAge' , 'CntFloorsArea' , 'CntFloorsHeight' , 'AreaPerAge' , 'HeightPerAge' , 'AreaPerHeight' , 'CntFamFloors' ,  'CntFamArea' ,  'CntFamHeight']
                    
        dummies = ['count_families_0','count_families_1','count_families_2',
                        'count_floors_pre_eq_1','count_floors_pre_eq_2',
                        'foundation_type_1', 'ground_floor_type_1',
                        'land_surface_condition_t','land_surface_condition_n',
                        'other_floor_type_j','other_floor_type_q', 'other_floor_type_x', 
                        'position_s','position_t',
                        'roof_type_n','roof_type_q']
                        
        binary = ['has_secondary_use',
                'has_secondary_use_agriculture',
                'has_superstructure_adobe_mud',
                'has_superstructure_cement_mortar_brick',
                'has_superstructure_mud_mortar_brick',
                'has_superstructure_mud_mortar_stone',
                'has_superstructure_timber']


        #Probabilities
        self.probabilities(1)
        self.probabilities(2)
        self.probabilities(3)

        df= self.df
        test = self.test
        
        #New features:
        df['CntFloorAge'] = df['count_floors_pre_eq']/(df['age']+0.1)
        df['CntFloorsArea'] = df['count_floors_pre_eq']/df['area_percentage']
        df['CntFloorsHeight'] = df['count_floors_pre_eq']/df['height_percentage']
        df['AreaPerAge'] = df['area_percentage']/(df['age']+0.1)
        df['HeightPerAge'] = df['height_percentage']/(df['age']+0.1)
        df['AreaPerHeight'] = df['area_percentage']/df['height_percentage']
        df['CntFamFloors'] = df['count_families']/df['count_floors_pre_eq']
        df['CntFamArea'] = df['count_families']/df['area_percentage']
        df['CntFamHeight'] = df['count_families']/df['height_percentage']

        test['CntFloorAge'] = test['count_floors_pre_eq']/(test['age']+0.1)
        test['CntFloorsArea'] = test['count_floors_pre_eq']/test['area_percentage']
        test['CntFloorsHeight'] = test['count_floors_pre_eq']/test['height_percentage']
        test['AreaPerAge'] = test['area_percentage']/(test['age']+0.1)
        test['HeightPerAge'] = test['height_percentage']/(test['age']+0.1)
        test['AreaPerHeight'] = test['area_percentage']/test['height_percentage']
        test['CntFamFloors'] = test['count_families']/test['count_floors_pre_eq']
        test['CntFamArea'] = test['count_families']/test['area_percentage']
        test['CntFamHeight'] = test['count_families']/test['height_percentage']

            #Powertransform

        pt1 = PowerTransformer(method= 'yeo-johnson')
        pt2 = PowerTransformer(method= 'yeo-johnson')
        pt3 = PowerTransformer(method= 'yeo-johnson')

        pt1.fit(df[['age']])
        pt2.fit(df[['area_percentage']])
        pt3.fit(df[['height_percentage']])

        df['age_pt'] = pt1.transform(df[['age']])
        df['area_percentage_pt'] = pt2.transform(df[['area_percentage']])
        df['height_percentage_pt'] = pt3.transform(df[['height_percentage']])

        test['age_pt'] = pt1.transform(test[['age']])
        test['area_percentage_pt'] = pt2.transform(test[['area_percentage']])
        test['height_percentage_pt'] = pt3.transform(test[['height_percentage_']])

        #Categorical

        df['count_families'] = df['count_families'].apply(lambda x: str(x) if x <3 else '+3')
        df['count_floors_pre_eq'] = df['count_floors_pre_eq'].apply(lambda x: str(x) if x <3 else '+3')
        df['position'] = df['position'].apply(lambda x: 'j' if x == 'o' else x)
        df['legal_ownership_status'] = df['legal_ownership_status'].apply(lambda x: str(1) if x == 'v' else str(0))
        df['foundation_type'] = df['foundation_type'].apply(lambda x: str(1) if x == 'r' else str(0))
        df['ground_floor_type'] = df['ground_floor_type'].apply(lambda x: str(1) if x == 'f' else str(0))
        df['plan_configuration'] = df['plan_configuration'].apply(lambda x : str(1) if x=='d' else str(0))

        test['count_families'] = test['count_families'].apply(lambda x: str(x) if x <3 else '+3')
        test['count_floors_pre_eq'] = test['count_floors_pre_eq'].apply(lambda x: str(x) if x <3 else '+3')
        test['position'] = test['position'].apply(lambda x: 'j' if x == 'o' else x)
        test['legal_ownership_status'] = test['legal_ownership_status'].apply(lambda x: str(1) if x == 'v' else str(0))
        test['foundation_type'] = test['foundation_type'].apply(lambda x: str(1) if x == 'r' else str(0))
        test['ground_floor_type'] = test['ground_floor_type'].apply(lambda x: str(1) if x == 'f' else str(0))
        test['plan_configuration'] = test['plan_configuration'].apply(lambda x : str(1) if x=='d' else str(0))


        #Imputer

        test[numeric] = KNNImputer(n_neighbors = 12).fit_transform(test[numeric])
        df[numeric] = KNNImputer(n_neighbors = 12).fit_transform(df[numeric])


        #Clustering
        clus_train = df[numeric+ [target]]
        clus_test = test[numeric]

        km = KMeansFeaturizer(k=12, target_scale =0.5, random_state= 1995)

        km.fit(clus_train, clus_test)

        clus_train = km.transform(clus_train[numeric])
        clus_test = km.transform(clus_test[numeric])

        clus_train = [clus_train[i][0] for i in range(len(clus_train))]
        clus_test = [clus_test[i][0] for i in range(len(clus_test))]
        df['clusters'] = clus_train
        test['clusters'] = clus_test


        self.df = pd.get_dummies(df)
        self.test = pd.get_dummies(test)

            

    def predict(self,df,test):       
        lgbm_model = LGBMClassifier(learning_rate=0.01, n_estimators=425, num_leaves=135,objective='multiclass', random_state=seed)
        lr_model.fit(self.df.drop('grade_damage',axis =1), self.df['grade_damage'])
        pred =lr_model.predict(self.test.drop('building_id',axis =1))

        return pred