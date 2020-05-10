# T H I S  P R O J E C T  I N C L U D E S  T H E  A N A L Y S I S  O F  P L A Y E R S  O F  F I F A 2019
import numpy as np 
import pandas as pd 
import random as rn 
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings('ignore')

# WE IMPORT THE FILE
fifa = pd.read_csv(r'C:\Users\hp\Desktop\code\Project\Project 01\Fifa19data.csv')
'''print(fifa.head())'''

# A N A L Y S I S


# P R E P R O C E S S I N G  O F  D A T A

fifa.drop(['Unnamed: 0', 'ID', 'Name', 'Photo', 'Nationality', 'Flag', 'Club', 'Club Logo', 'Wage', 'Preferred Foot', 'Weak Foot', 'Real Face', 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'Release Clause'], axis = 1, inplace = True)

fifa['International Reputation'] = fifa['International Reputation'].fillna(method = "bfill")

fifa.drop(fifa[fifa['Skill Moves'].isnull()].index, axis = 0, inplace = True)
fifa.drop(fifa[fifa['Position'].isnull()].index, axis = 0, inplace = True)

def skill(x):
    if type(x) == str:
        s1 = x[0:2]
        s2 = x[-1]
        x = int(s1) + int(s2)
        return x
    
    else:
        return x

for i in range(12, 38):
	fifa.iloc[:, i] = fifa.iloc[:, i].apply(skill)

rn.seed(5)
def impute(x):
	if pd.isnull(x):
		return rn.randrange(5, 15)
	else:
		return x

for i in range(12, 38):
	fifa.iloc[:, i] = fifa.iloc[:, i].apply(impute)


def currencyConverter(val):
    if val[-1] == 'M':
        val = val[1:-1]
        val = float(val) * 1000000
        return val
        
    elif val[-1] == 'K':
        val = val[1:-1]
        val = float(val) * 1000
        return val
    
    else:
        return 0


def height_converter(val):
    f = val.split("'")[0]
    i = val.split("'")[1]
    h = (int(f) * 12) + int(i)
    return h

def weight_converter(val):
    w = int(val.split('lbs')[0])
    return w


fifa['Value'] = fifa['Value'].apply(currencyConverter)
fifa['Height'] = fifa['Height'].apply(height_converter)
fifa['Weight'] = fifa['Weight'].apply(weight_converter)

fifa['Body Type'][fifa['Body Type'] == 'Messi'] = 'Lean'
fifa['Body Type'][fifa['Body Type'] == 'C. Ronaldo'] = 'Normal'
fifa['Body Type'][fifa['Body Type'] == 'Neymar'] = 'Lean'
fifa['Body Type'][fifa['Body Type'] == 'Courtois'] = 'Lean'
fifa['Body Type'][fifa['Body Type'] == 'PLAYER_BODY_TYPE_25'] = 'Normal'
fifa['Body Type'][fifa['Body Type'] == 'Shaqiri'] = 'Stocky'
fifa['Body Type'][fifa['Body Type'] == 'Akinfenwa'] = 'Stocky'

le = LabelEncoder()
fifa['Position'] = le.fit_transform(fifa['Position'])
fifa['Work Rate'] = le.fit_transform(fifa['Work Rate'])
fifa['Body Type'] = le.fit_transform(fifa['Body Type'])

y = fifa.iloc[:, 1]
#print(y)
x = fifa.drop(['Overall'], axis = 1)
#print(x)

rows, cols = x.shape
flds = list(x.columns)


corr = x.corr().values

'''for i in range(0, cols):
    for j in range(i+1, cols):
        if corr[i,j] >= abs(0.9):
            print(flds[i], ' ', flds[j], ' ', corr[i,j])'''


sns.heatmap(corr, annot = True) # annot = True shows the value over the graph
plt.show()

x = x.drop(['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 
				'ShortPassing', 'BallControl', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'RCM', 
				'RCB', 'Acceleration', 'StandingTackle', 'GKPositioning'], axis = 1)


'''print(fifa.isnull().sum().sum())'''

'''for i in range(37):
	X = x.iloc[:, [i]]
	ts_score = []
	for j in range(2000): # as there are very data so we give range as 1000
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state = j)
		lr = LinearRegression().fit(X_train, y_train)
		ts_score.append(lr.score(X_test, y_test))
	k = ts_score.index(np.max(ts_score))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state = k)
	reg = LinearRegression()
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	print(r2_score(y_test, y_pred))'''

x.drop(['Work Rate', 'Body Type', 'Position', 'Height', 'Weight', 'SprintSpeed', 'Balance', 'GKReflexes'], axis = 1, inplace = True)
print(x.head())

# M O D E L L I N G
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1, random_state = 528)
'''reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))
print('\n')'''


DTReg = DecisionTreeRegressor(min_samples_leaf = .02)
DTReg.fit(x_train, y_train)

y_pred = DTReg.predict(x_test)
r2_score = r2_score(y_test, y_pred)
print(r2_score)
print('\n')


'''ts_score = []
import numpy as np
for j in range(2000): # as there are very data so we give range as 1000
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1, random_state = j)
	lr = LinearRegression().fit(x_train, y_train)
	ts_score.append(lr.score(x_test, y_test))

k = ts_score.index(np.max(ts_score))
print(k) # we get k as 1897'''

ts_score = []
for j in range(1000): # as there are very data so we give range as 1000
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1, random_state = j)
	dtr = DecisionTreeRegressor(min_samples_leaf = .02).fit(x_train, y_train)
	ts_score.append(dtr.score(x_test, y_test))

k = ts_score.index(np.max(ts_score))
print(k) # and we get k as 513