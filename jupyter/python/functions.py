#Functions

#        DATA
# ==================== #
import pandas as pd
import numpy as np
from scipy import stats
import time
import random
import math



#      PLOTING
# ============================== #

from PIL import Image
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme(style="white")




def flatten(t): 
    """
    Utilizamos la funcion para apilar listas
    """
    return [item for sublist in t for item in sublist]

def boxplots_algorithms(results, names):
    """ 
    For plot the results
    
    Para plotear los resultados 
    Input
    --------
    results (Pandas DF): Results of training models
    names: Names of the models

    Output
    --------
    Image in the notebook
    """
    
    plt.figure(figsize=(8,8))
    plt.boxplot(results)
    plt.xticks(range(1,len(names)+1), names)
    plt.show()


def img_reshape(img):
    """
    For show a image.
    Para mostrar una imagen.

    Input
    --------
    img (String): path image

    Output
    --------
    Image in the notebook

    """
    img = Image.open('./images/'+img).convert('RGB')
    img = img.resize((300,500))
    img = np.asarray(img)
    return img

def img_reshape_more(img):
    """
    For show a image.
    Para mostrar una imagen.

    Input
    --------
    img (String): path image

    Output
    --------
    Image in the notebook

    """
    img = Image.open('./images/'+img).convert('RGB')
    img = img.resize((1000,500))
    img = np.asarray(img)
    return img


def probabilities(df, test, n): 
    """
    Calculate probabilities for distinc geo level and target groups.
    Calcular las probabilidades para los distintos geo level y grupos de la target.

    Input
    --------
    df (DataFrame pandas)
    n (Int): values 1,2,3 for distinct geo level

    Output
    --------
    Dataframe with probabilities new columns
    
    """
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

    df[nom1[0]] = list1
    df[nom2[0]] = list2
    df[nom3[0]] = list3

    list1 = []
    list2 = []
    list3 = []

    for i in test[column[0]]:
        list1.append(damage1.get(i))
        list2.append(damage2.get(i))
        list3.append(damage3.get(i))

    test[nom1[0]] = list1
    test[nom2[0]] = list2
    test[nom3[0]] = list3
    
    return df , test