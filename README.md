# Model 1 - Ty Pennington
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. What this data set does not account for is what I believe to be one of the most valuable determinants of price, "beauty lies in the eye of the beholder." In other words, the value of a house is often subjective - what one person values at 1,000,000 another may value at 875,000

With 79 explanatory variables describing (almost) every aspect of residential homes in (INSERT TOWN HERE), this data set challenges us to predict the final price of each home.

Ty Pennington is an XGBoost predictive model built to predict residential housing prices in Ames Iowa. XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost has become a widely used and really popular tool among Kaggle competitors and Data Scientists in industry, as it has been battle tested for production on large-scale problems. It is a highly flexible and versatile tool that can work through most regression, classification and ranking problems as well as user-built objective functions. 

### Handling Missing Values
There are many ways data can end up with missing values. For example, a 2 bedroom house would not include a data point indicating the size of a third bedroom. A homeowner being surveyed may choose not to share specific information regarding their property. Python libraries represent missing numbers as NaN which is short for "not a number". Most libraries (including scikit-learn) will give you an error if you try to build a model using data with missing values.

In general, one can either drop columns with missing values or impute missing values. Dropping columns entirely can be useful when most values in a column are missing. Imputation fills in the missing value with some number. The imputed value won't be exactly right, however, it helps to produce more accurate predictive models. I've developed a few strategies to intuitively handle said missing values.

```
# Calculating percentage of missing values (per feature)
nan = (df.isnull().sum() / len(df)) * 100
nan = nan.drop(nan[nan == 0].index).sort_values(ascending=False)
missing_percentage = pd.DataFrame({'% of Missing Values' : nan})
missing_percentage.head(20)

# Visualizing the percentages calculated above
f, ax = plt.subplots(figsize=(10, 10))
plt.xticks(rotation='90')
nan = nan.sort_values()
sns.barplot(x=nan.index, y=nan)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percentage', fontsize=15)
plt.title('Percent of Missing Values by Feature', fontsize=15);
```
<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/missingvaluespercentages.PNG?raw=true" width="500" height="500" />

```
# Imputing missing pools
df["PoolQC"] = df["PoolQC"].fillna("None")

# Imputing MiscFeature 
df["MiscFeature"] = df["MiscFeature"].fillna("None")

# Imputung Alley 
df["Alley"] = df["Alley"].fillna("None")

# Imputing Fence
df["Fence"] = df["Fence"].fillna("None")

# Imputing FireplaceQu
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

# Imputing garage information
for column in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'): df[column] = df[column].fillna('None')
for column in ('GarageYrBlt', 'GarageArea', 'GarageCars'): df[column] = df[column].fillna(0)

# Imputing basement information
for column in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'): df[column] = df[column].fillna(0)
for column in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'): df[column] = df[column].fillna('None')

# Imputing masonary work
df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

# Imputing zoning
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

# Imputing Functional
df["Functional"] = df["Functional"].fillna("Typ")

# Imputing Electrical
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Imputing KitchenQual
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

# Imputing Exterior1st and Exterior2nd
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

# Imputing SaleType
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

# Imputing MSSubClass 
df['MSSubClass'] = df['MSSubClass'].fillna("None")

# Imputing LotFrontage --> Calculating median neighborhood LotFrontage
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
```
### Feature Engineering
The goal of feature engineering is simply to make your data better suited to the problem at hand. For a feature to be useful, it must have a relationship to the target that your model is able to learn. For example, linear models are only able to learn linear relationships. Therefore, when building a linear model, your goal is to transform input features so that their relationship to the target becomes linear. Common benefits of feature engineering include improved predictive performance, reduced computational needs and improved interpretability of results.
```
# Some of the non-numeric predictors are stored as numbers --> convert them into strings 
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['OverallCond'] = df['OverallCond'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
```
