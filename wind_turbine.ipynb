import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
wind_turbine=pd.read_csv(r'Wind_Turbine_2025.csv')
wind_turbine.info()
# First_bussiness_moment
wind_turbine.describe()

# 
Extracting Numeric Colunms from the data 
numeric_columns=wind_turbine.select_dtypes(include=np.number)
numeric_columns.head()
numeric_columns.median()

# Second_Bussiness_Moment
numeric_columns.var()
ranges = numeric_columns.max() - numeric_columns.min()
ranges
# Third_Bussiness_Moment
numeric_columns.skew() 
# Fourth_Bussiness_Moment
numeric_columns.kurtosis()
# Graphical Representation
# Histogram
wind_turbine.hist(color='skyblue', edgecolor='black', bins=20, figsize=(14, 12))
# Boxplots
plt.figure(figsize=(14, 8))
wind_turbine.boxplot()
plt.xticks(rotation=45)
plt.show()
# line charts 
wind_turbine.head()
# corelation

numeric_columns.corr()
sns.scatterplot(data=wind_turbine, x='Wind_speed', y='Power')
plt.title('Scatter Plot: Wind Speed vs Power')
plt.xlabel('Wind Speed')
plt.ylabel('Power')
plt.show()
wind_turbine['date']=wind_turbine['date'].astype('datetime64[ns]')
wind_turbine.info()
sns.lineplot(data=wind_turbine.head(100), x='date', y='Power')
plt.show()
sns.lineplot(data=wind_turbine.head(100), x='date', y='Wind_speed')
plt.show()
sns.scatterplot(data=wind_turbine, x='Power', y='Rotor_Speed')
plt.title('Scatter Plot: Power vs Rotor_Speed')
plt.xlabel('power')
plt.ylabel('Rotor_Speed	')
plt.show()
# Data_Preprocessing 
wind_turbine.shape
# Type casting
wind_turbine.head()
wind_turbine.info()
# changing the date in object to the datetime[ns]
wind_turbine['date']=wind_turbine['date'].astype('datetime64[ns]')
wind_turbine.info()
# Handling_missing_values 
wind_turbine.isna().sum()
# we need to verify the distribution of the data first then if the distribution is normal then use mean imputation(for nrl the outliers are low so we use mean)
# if the distriibution is skewed then median imputation the data is categorical we use categorical we use mode imputation(the medain is not effected by outliers)
# when we know that the values are actually missing then we use constants like 0 or not defined or not available 
# we need not to change the distribution then we use the random distribution 
fig = px.histogram(
    wind_turbine,
    x='Wind_speed',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Wind Speed'
)
fig.show()
numeric_colunms=wind_turbine.select_dtypes(include=np.number)
numeric_colunms.mean()
numeric_colunms.median()
wind_turbine[['Wind_speed']].skew()
from sklearn.impute import SimpleImputer 
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Wind_speed']]=median_imputer.fit_transform(wind_turbine[['Wind_speed']])
print("missing values:",wind_turbine['Wind_speed'].isna().sum())
fig = px.histogram(
    wind_turbine,
    x='Wind_speed',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Wind Speed'
)
fig.show()
# Outlier Treatment 
# we use capping method for the numeric and extreme outliers 
# we use removing for the less outliers 
wind_turbine[['Wind_speed']].boxplot()
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        tail = 'both', 
                        fold = 1.5, 
                        variables = ['Wind_speed'])
wind_turbine['Wind_speed']= winsor_iqr.fit_transform(wind_turbine[['Wind_speed']])
wind_turbine[['Wind_speed']].boxplot()
fig = px.histogram(
    wind_turbine,
    x='Wind_speed',
    nbins=50,
    marginal='box',  
    histnorm='probability density', 
    title='Distribution of Wind Speed'
)
fig.show()
wind_turbine[['Wind_speed']].skew()
wind_turbine[['Wind_speed']].kurt()
wind_turbine[['Wind_speed']].mean()
wind_turbine[['Wind_speed']].median()
from feature_engine.transformation import YeoJohnsonTransformer 
fitted = YeoJohnsonTransformer(variables=['Wind_speed']) 
fitted_data = fitted.fit_transform(wind_turbine)
fig = px.histogram(
    wind_turbine,
    x='Wind_speed',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Wind Speed'
)
fig.show()
# The data is not normally distributed then we use 
# if data is numeric or normal then use the StandardScaler
# if data is skewed then use minmaxscaler(Scales to [0, 1])
# if data is categorical then use one hot encoding or label encoding 
# Centers to mean=0, std=1 then standardscalar
# Data has outliers(robust scaling)
# StandardScaler:Centers data around the mean (0) and scales using standard deviation.Works best for normally distributed data.
#                Common in regression, SVM, and PCA models.
# MinMaxScaler:Rescales data to a fixed range [0, 1] (or custom range).
#             Keeps the original shape of the distribution.
#              Ideal for neural networks and algorithms sensitive to feature magnitude.
# RobustScaler:Centers data around the median (0) and scales using the IQR (Q3–Q1).
#              Reduces the effect of outliers.
#              Preferred when the dataset contains extreme or skewed values.

from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
wind_turbine[['Wind_speed']] = robust_scaler.fit_transform(wind_turbine[['Wind_speed']])

fig = px.histogram(
    wind_turbine,
    x='Wind_speed',
    nbins=50,
    marginal='box',  
    histnorm='probability density', 
    title='Distribution of Wind Speed'
)
fig.show()
wind_turbine['Wind_speed'].skew()
wind_turbine['Wind_speed'].kurt()

# Data preprocessing for the power colunm
fig = px.histogram(
    wind_turbine,
    x='Power',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Power'
)
fig.show()
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Power']]=median_imputer.fit_transform(wind_turbine[['Power']])
                            
print("missing values:",wind_turbine['Power'].isna().sum())
fig = px.histogram(
    wind_turbine,
    x='Power',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Power'
)
fig.show()

from feature_engine.outliers import Winsorizer
winsor_quantiles = Winsorizer(capping_method='quantiles', 
                        tail='both', 
                        fold=0.02, 
                        variables=['Power'])
wind_turbine[['Power']]=winsor_quantiles.fit_transform(wind_turbine[['Power']])
fig = px.histogram(
    wind_turbine,
    x='Power',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Power'
)
fig.show()
wind_turbine[['Power']].skew()
wind_turbine[['Power']].boxplot()
wind_turbine['Power'].skew()
wind_turbine['Power'].kurt()
# Data is right-skewed and positive only -log 
# Data can be positive or negative - YeoJohnson
# Data is positive only -boxcox 
from feature_engine.transformation import YeoJohnsonTransformer
yj = YeoJohnsonTransformer(variables=['Power'])
wind_turbine = yj.fit_transform(wind_turbine)
fig = px.histogram(
    wind_turbine,
    x='Power',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Power'
)
fig.show()
wind_turbine['Power'].skew()
wind_turbine['Power'].kurt()
wind_turbine.isna().sum()
# Data preprocessing for Nacelle_ambient_temperature
fig = px.histogram(
    wind_turbine,
    x='Nacelle_ambient_temperature',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Nacelle_ambient_temperature'
)
fig.show()
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Nacelle_ambient_temperature']]=median_imputer.fit_transform(wind_turbine[['Nacelle_ambient_temperature']])
wind_turbine[['Nacelle_ambient_temperature']].skew()
fig = px.histogram(
    wind_turbine,
    x='Nacelle_ambient_temperature',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Nacelle_ambient_temperature'
)
fig.show()
wind_turbine[['Nacelle_ambient_temperature']].boxplot()
wind_turbine[['Nacelle_ambient_temperature']].skew()
#  Data preprocessing for Generator_bearing_temperature
fig = px.histogram(
    wind_turbine,
    x='Generator_bearing_temperature',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Nacelle_ambient_temperature'
)
fig.show()
wind_turbine[['Generator_bearing_temperature']].skew()
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Generator_bearing_temperature']]=median_imputer.fit_transform(wind_turbine[['Generator_bearing_temperature']])
print("missing values:",wind_turbine['Generator_bearing_temperature'].isna().sum())
wind_turbine[['Generator_bearing_temperature']].boxplot()
wind_turbine[['Generator_bearing_temperature']].skew()
wind_turbine[['Generator_bearing_temperature']].kurt()
# Data preprocessing for Gear_oil_temperature
wind_turbine[['Gear_oil_temperature']].isna().sum()
sns.histplot(wind_turbine['Gear_oil_temperature'],kde='True')
wind_turbine[['Gear_oil_temperature']].skew()
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Gear_oil_temperature']]=median_imputer.fit_transform(wind_turbine[['Gear_oil_temperature']])
sns.histplot(wind_turbine['Gear_oil_temperature'],kde='True')
wind_turbine[['Gear_oil_temperature']].boxplot()
wind_turbine[['Gear_oil_temperature']].skew()
wind_turbine[['Gear_oil_temperature']].kurt()
print(wind_turbine.columns)
# Data Preprocessing for the  Ambient_temperature
 fig = px.histogram(
    wind_turbine,
    x='Ambient_temperature',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Ambient_temperature'
)
fig.show()
wind_turbine['Ambient_temperature'].isna().sum()
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Ambient_temperature']]=median_imputer.fit_transform(wind_turbine[['Ambient_temperature']])
wind_turbine['Ambient_temperature'].isna().sum()
 fig = px.histogram(
    wind_turbine,
    x='Ambient_temperature',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Ambient_temperature'
)
fig.show()
wind_turbine[['Ambient_temperature']].boxplot()
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method='iqr', 
                        tail='both', 
                        fold=1.5, 
                        variables=['Ambient_temperature'])
wind_turbine[['Ambient_temperature']]=winsor_iqr.fit_transform(wind_turbine[['Ambient_temperature']])
wind_turbine[['Ambient_temperature']].boxplot()
fig = px.histogram(
    wind_turbine,
    x='Ambient_temperature',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Ambient_temperature'
)
fig.show()
wind_turbine['Ambient_temperature'].skew()
from feature_engine.transformation import YeoJohnsonTransformer
yj = YeoJohnsonTransformer(variables=['Ambient_temperature'])
wind_turbine = yj.fit_transform(wind_turbine)
wind_turbine['Ambient_temperature'].skew()
fig = px.histogram(
    wind_turbine,
    x='Ambient_temperature',
    nbins=50,
    marginal='box',   
    histnorm='probability density', 
    title='Distribution of Ambient_temperature'
)
fig.show()
wind_turbine['Ambient_temperature'].skew()
wind_turbine.isna().sum()
# Data preprocessing for Rotor_Speed 
sns.histplot(wind_turbine['Rotor_Speed'],kde=True) 
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Rotor_Speed']]=median_imputer.fit_transform(wind_turbine[['Rotor_Speed']])
wind_turbine['Rotor_Speed'].skew()
sns.histplot(wind_turbine['Rotor_Speed'],kde=True) 
wind_turbine[['Rotor_Speed']].boxplot()
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method='iqr', 
                        tail='both', 
                        fold=1.5, 
                        variables=['Rotor_Speed'])
wind_turbine[['Rotor_Speed']]=winsor_iqr.fit_transform(wind_turbine[['Rotor_Speed']])
wind_turbine[['Rotor_Speed']].boxplot()
wind_turbine['Rotor_Speed'].skew()
sns.histplot(wind_turbine['Rotor_Speed'],kde=True) 
wind_turbine['Rotor_Speed'].kurt()
# Data Preprocessing for the Nacelle_temperature
wind_turbine.isna().sum()
sns.histplot(wind_turbine['Nacelle_temperature'],kde=True)
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Nacelle_temperature']]=median_imputer.fit_transform(wind_turbine[['Nacelle_temperature']])
wind_turbine['Nacelle_temperature'].isna().sum()
wind_turbine['Nacelle_temperature'].skew()
wind_turbine[['Nacelle_temperature']].boxplot()
wind_turbine['Nacelle_temperature'].kurt()
wind_turbine['Nacelle_temperature'].skew()
sns.histplot(wind_turbine['Nacelle_temperature'],kde=True)
wind_turbine.isna().sum()
sns.histplot(wind_turbine['Bearing_temperature'],kde=True)
wind_turbine['Bearing_temperature'].skew()
wind_turbine['Bearing_temperature'].kurt()
from sklearn.impute import SimpleImputer 
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
wind_turbine[['Bearing_temperature']]=mean_imputer.fit_transform(wind_turbine[['Bearing_temperature']])
sns.histplot(wind_turbine['Bearing_temperature'],kde=True)
wind_turbine['Bearing_temperature'].skew()
from sklearn.impute import SimpleImputer 
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Bearing_temperature']]=mean_imputer.fit_transform(wind_turbine[['Bearing_temperature']])
sns.histplot(wind_turbine['Bearing_temperature'],kde=True)
wind_turbine[['Bearing_temperature']].boxplot()
# Data Preprocessing for Generator_speed
sns.histplot(wind_turbine['Generator_speed'],kde=True)
from sklearn.impute import SimpleImputer 
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Generator_speed']]=median_imputer.fit_transform(wind_turbine[['Generator_speed']])
wind_turbine['Generator_speed'].skew()
wind_turbine['Generator_speed'].isna().sum()
sns.histplot(wind_turbine['Generator_speed'],kde=True)
from feature_engine.transformation import YeoJohnsonTransformer 
fitted = YeoJohnsonTransformer(variables=['Generator_speed']) 
fitted_data = fitted.fit_transform(wind_turbine)
sns.histplot(wind_turbine['Generator_speed'],kde=True)
from feature_engine.outliers import Winsorizer
winsor_quantiles = Winsorizer(capping_method='quantiles', 
                        tail='both', 
                        fold=0.02, 
                        variables=['Generator_speed'])
wind_turbine[['Generator_speed']]=winsor_quantiles.fit_transform(wind_turbine[['Generator_speed']])
sns.histplot(wind_turbine['Generator_speed'],kde=True)
wind_turbine['Generator_speed'].skew()
wind_turbine.isna().sum()
# Data Preprocessing for Yaw_angle 
sns.histplot(wind_turbine['Yaw_angle'],kde=True)
wind_turbine['Yaw_angle'].mean()
wind_turbine['Yaw_angle'].median()
wind_turbine['Yaw_angle'].skew()
wind_turbine['Yaw_angle'].kurt()
from sklearn.impute import SimpleImputer 
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
wind_turbine[['Yaw_angle']]=median_imputer.fit_transform(wind_turbine[['Yaw_angle']])
sns.histplot(wind_turbine['Yaw_angle'],kde=True)
wind_turbine[['Yaw_angle']].boxplot()
# Data Preprocessing for Wind_direction 
sns.histplot(wind_turbine['Wind_direction'],kde=True)
wind_turbine.isna().sum()
sns.histplot(wind_turbine['Wind_direction'], kde=True)

print(wind_turbine['Wind_direction'].mean())
print(wind_turbine['Wind_direction'].median())
print(wind_turbine['Wind_direction'].skew())
print(wind_turbine['Wind_direction'].kurt())
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

wind_turbine[['Wind_direction']] = median_imputer.fit_transform(wind_turbine[['Wind_direction']])
sns.histplot(wind_turbine['Wind_direction'], kde=True)
wind_turbine[['Wind_direction']].boxplot()
# Data Preprocessing for Wheel_hub_temperature 
sns.histplot(wind_turbine['Wheel_hub_temperature'], kde=True)
print(wind_turbine['Wheel_hub_temperature'].mean())
print(wind_turbine['Wheel_hub_temperature'].median())
print(wind_turbine['Wheel_hub_temperature'].skew())
print(wind_turbine['Wheel_hub_temperature'].kurt())
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

wind_turbine[['Wheel_hub_temperature']] = median_imputer.fit_transform(wind_turbine[['Wheel_hub_temperature']])

wind_turbine[['Wheel_hub_temperature']].boxplot()
sns.histplot(wind_turbine['Wheel_hub_temperature'], kde=True)
# Data Preprocessing for Gear_box_inlet_temperature 
sns.histplot(wind_turbine['Gear_box_inlet_temperature'], kde=True)
print(wind_turbine['Gear_box_inlet_temperature'].mean())
print(wind_turbine['Gear_box_inlet_temperature'].median())
print(wind_turbine['Gear_box_inlet_temperature'].skew())
print(wind_turbine['Gear_box_inlet_temperature'].kurt())
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

wind_turbine[['Gear_box_inlet_temperature']] = median_imputer.fit_transform(wind_turbine[['Gear_box_inlet_temperature']])
wind_turbine[['Gear_box_inlet_temperature']].boxplot()
sns.histplot(wind_turbine['Gear_box_inlet_temperature'], kde=True)
wind_turbine.isna().sum()
