# Model 1 - Ty Pennington
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. This dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. With 79 explanatory variables describing (almost) every aspect of residential homes in Groveland MA, this data set challenges us to predict the final price of each home. However, what this data set does not account for is what I believe to be one of the most valuable determinants of price, "beauty lies in the eye of the beholder." In other words, the value of a house is often subjective - what one person values at 1,000,000 another may value at 875,000

Ty Pennington is an XGBoost predictive model trained to predict residential housing prices in Groveland MA. XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost has become a widely used and really popular tool among Kaggle competitors and Data Scientists, as it has been battle tested for production on large-scale problems. It is a highly flexible and versatile tool that can work through most regression, classification and ranking problems as well as user-built objective functions. 

Move that bus!

---

## Part 1 - Data Cleaning
Does my data contain outliers? What should I do about missing values? Is my data skewed? Why aren’t my dates formatted correctly? Data cleaning answers these questions and provides intuitive solutions. Arguably the most important step in the data science process, data cleaning can easily become deeply frustrating and time intensive. Recent studies suggest that data scientists spend anywhere from 50% to 80% of their time cleaning data rather than creating insights.

### Handling Missing Values
Python libraries represent missing numbers as NaN which is short for "not a number". Most libraries (including scikit-learn) will give you an error if you try to build a model using data with missing values. In general, one can either drop columns with missing values or impute missing values. Dropping columns entirely can be useful when most values in a column are missing. Imputation fills in the missing value with some number. The imputed value won't be exactly right, however, it helps to produce more accurate predictive models.

<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/percentofmissingvalues.PNG?raw=true" width="500" height="300" />

```python
# Miscellaneous features (None)
df["PoolQC"] = df["PoolQC"].fillna("None")
df["MiscFeature"] = df["MiscFeature"].fillna("None")
df["Alley"] = df["Alley"].fillna("None")
df["Fence"] = df["Fence"].fillna("None")
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
df['MSSubClass'] = df['MSSubClass'].fillna("None")
df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["Functional"] = df["Functional"].fillna("Typ")
```
```python
# Miscellaneous features (mode)
df["MSZoning"] = df["MSZoning"].fillna(df["MSZoning"].mode()[0])
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
df["KitchenQual"] = df["KitchenQual"].fillna(df["KitchenQual"].mode()[0])
df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
```
```python
# Garage features
for column in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"): df[column] = df[column].fillna("None")
for column in ("GarageYrBlt", "GarageArea", "GarageCars"): df[column] = df[column].fillna(0)

# Basement features
for column in ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"): df[column] = df[column].fillna(0)
for column in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"): df[column] = df[column].fillna('None')
```
```python
# Converting numerical features to categorical features
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['OverallCond'] = df['OverallCond'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
```
## Part 2 - Iterative Modeling

### Baseline w/ 10-Fold Cross-validation
Using a naive model we can establish a baseline score that will allow us to monitor our model's predictive performance following additional data processing and feature engineering. As we iteratively cycle through efforts to improve our model we will observe `MSE`, `RMSE` and `RMSLE`. Mean squared error measures the average of the squares of our model's errors — that is, the average squared difference between the estimated values and the actual values. `RMSE` simply calculates the root of `MSE` while `RMSLE` calculate the log root of MSE. It is important to note that these metrics are quite sensitive to outliers.

* `Cross-validation` is a variation of resampling that can be used to evaluate machine learning models on a limited data sample. The cross-validation modeling process is run on different subsets of our data to get multiple (out of sample) measures of model quality. For example, consider a 10 fold modeling process. We would divide our data into 10 pieces, each representing 10% of our full dataset. Experiment 1 would be run using the first fold as a holdout or test set, while everything else would be considered training data. This provides a measure of model quality based on a 10% holdout set. This process is repeated until every fold is used once as the holdout set.

```python
# Creating a reusable function to calculate RMSE & RMSLE
def evaluate(data, model):

  # Initializing X and y
  X = data.copy()
  y = X.pop("SalePrice")
  log_y = np.log(y)

  # Initializing an encoder
  ohe = ce.OneHotEncoder()
  X = ohe.fit_transform(X)

  # RMSE 10-fold cross validation
  kfold = KFold(n_splits=10,shuffle=True,random_state=42)
  mse = -1 * cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kfold)
  rmse = np.sqrt(mse)
  
  # RMSLE 10-fold cross validation
  kfold = KFold(n_splits=10,shuffle=True,random_state=42)
  msle = -1 * cross_val_score(model, X, log_y, scoring="neg_mean_squared_error", cv = kfold)
  rmsle = np.sqrt(msle)
  return rmse, rmsle
```
```
RMSE = 28312.99466499266
RMSLE = 0.13216266198134258
```

### Outliers
An outlier is an observation that diverges from an overall pattern within a sample. Mathematically, an outlier is usually defined as an observation more than three standard deviations from the mean (although sometimes you'll see 2.5 or 2 as well). Most machine learning algorithms do not work well in the presence of outliers, as they are known to skew mean and standard deviation, reduce the effectiveness of statistical tests and decrease normality.

Because of the nuances of each data set, there is no precise way to define outliers in general because of the specifics of each data set. However, there are multiple methods that can be used to identify "potential" otliers. As the domain experts, we must interpret the observations and decide whether a value is an outlier or not. Said identification methods include Z-scores, Robust Z-scores, I.Q.R measurements and basic visualizations.

<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/SalePriceOutliersWithCutoff.PNG?raw=true" width="400" height="300" />

```python
# I've decided to delete observations with SalePrice > $550,000
df.drop(df[(df["SalePrice"]>550000)].index, inplace=True)
```
```
# Has our model improved? It sure has!
RMSE = 24438.453471065826
RMSLE = 0.13041266719439537
```

### Feature Engineering
Feature engineering is the process of extracting features or input variables from raw data which can be applied to our model to improve predictive performance and interpretability of results, while reducing computational needs. The process involves a combination of statistical analysis, domain knowledge and creativity.


```python
# LivLotRatio
train["LivLotRatio"] = train["GrLivArea"] / train["LotArea"]
test["LivLotRatio"] = test["GrLivArea"] / test["LotArea"]
```
```python
# TotalSf
train["TotalSF"] = train["TotalBsmtSF"] + train["1stFlrSF"] + train["2ndFlrSF"]
test["TotalSF"] = test["TotalBsmtSF"] + test["1stFlrSF"] + test["2ndFlrSF"]
```
```python
# AvgNeighborhoodGrLivArea
NeighborhoodGrLivArea = train.groupby("Neighborhood")[["GrLivArea"]].median().rename({"GrLivArea": "AvgNeighborhoodGrLivArea"}, axis=1)
train = train.merge(NeighborhoodGrLivArea, left_on="Neighborhood", right_index=True, how="left")
test = test.merge(NeighborhoodGrLivArea, left_on="Neighborhood", right_index=True, how="left")

# GrLivAreaVariance
train["GrLivAreaVariance"] = train["GrLivArea"] / train["AvgNeighborhoodGrLivArea"]
test["GrLivAreaVariance"] = test["GrLivArea"] / test["AvgNeighborhoodGrLivArea"]

# Drop AvgNeighborhoodGrLivArea
train.drop("AvgNeighborhoodGrLivArea", axis=1, inplace=True)
test.drop("AvgNeighborhoodGrLivArea", axis=1, inplace=True)
```
```python
# AvgNeighborhoodOverallQual
NeighborhoodOverallQual = train.groupby("Neighborhood")[["OverallQual"]].mean().rename({"OverallQual": "AvgNeighborhoodOverallQual"}, axis=1)
train = train.merge(NeighborhoodOverallQual, left_on="Neighborhood", right_index=True, how="left")
test = test.merge(NeighborhoodOverallQual, left_on="Neighborhood", right_index=True, how="left")

# OverallQualVariance
train["OverallQualVariance"] = train["OverallQual"] - train["AvgNeighborhoodOverallQual"]
test["OverallQualVariance"] = test["OverallQual"] - test["AvgNeighborhoodOverallQual"]

# Drop AvgNeighborhoodOverallQual
train.drop("AvgNeighborhoodOverallQual", axis=1, inplace=True)
test.drop("AvgNeighborhoodOverallQual", axis=1, inplace=True)
```
```
# Has our model improved? It sure has!
RMSE = 24457.020994094106
RMSLE = 0.1289761000009596
```






### Correlation
A statistical measurement of the degree to which two variables are linearly related. `Positive correlation` is a relationship between two variables in which both variables move in the same direction. This is when one variable increases while the other increases and visa versa. Whilst `negative correlation` is a relationship where one variable increases as the other decreases. Correlation can be extremely useful when experimenting with feature engineering.

```python
# Calculating pearson correlation
matrix = train.corr()
matrix["SalePrice"].sort_values(ascending=False).head(15)
```
```python
# Masking the upper triangle
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

# Visualizing correlation with a heatmap
f, ax = plt.subplots(figsize=(10, 10))
heatmap = sns.heatmap(corr_matrix,
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4,
                                "ticks" : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = False,
                      annot_kws = {"size": 12})

# Setting column names as labels
ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns)
ax.set_title("Feature Correlation to SalePrice")

# Setting styles
sns.set_style('whitegrid')
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
```

<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/correlation.PNG?raw=true" width="500" height="400" />

### Mutual Information
Mutual information is a great general-purpose metric and especially useful at the start of feature development. Mutual information is a lot like correlation in that it measures a relationship between two quantities. However, the advantage of mutual information is that it can detect any kind of relationship, while correlation only detects linear relationships. The mutual information between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. If you knew the value of a feature, how much more confident would you be about the target?

* It's possible for a feature to be very informative when interacting with other features, but not so informative all alone. MI can't detect interactions between features. It is a univariate metric.
* The actual usefulness of a feature depends on the model you use it with. A feature is only useful to the extent that its relationship with the target is one your model can learn. Just because a feature has a high MI score doesn't mean your model will be able to do anything with that information. You may need to transform the feature first to expose the association.

```python
# Creating a function to calculate MI scores
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
```
```python
# Creating a function to visualize MI scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
```


### Skewness & Kurtosis
Skewness measures the degree of distortion from the symmetrical bell curve or the normal distribution. A symmetrical or normal distribution will have a skewness of 0. Skewness between -0.5 and 0.5 is considered fairly symmetrical. Skewness between -1 and -0.5 or between 0.5 and 1 is considered moderately skewed. Skewness less than -1 or greater than 1 is considered highly skewed.

* `Positive Skewness` - when the tail on the right side of the distribution is longer or fatter. The mean and median will be greater than the mode.
* `Negative Skewness` - is when the tail of the left side of the distribution is longer or fatter than the tail on the right side. The mean and median will be less than the mode.

Kurtosis is used to describe/measure the extreme values in one versus the other tail. It is actually the measure of outliers present in the distribution. High kurtosis in a data set is an indicator that data has heavy tails or outliers. Low kurtosis in a data set is an indicator that data has light tails or lack of outliers.

```python
# Visualizing the distribution of our target variable
sns.distplot(df['SalePrice'], fit=norm);

# calculate the fitted parameters used by the function
(mu, sigma) = norm.fit(df['SalePrice'])

# Plotting the distribution
plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.ticklabel_format(style='plain');

# Visualize the QQ-plot
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
plt.show();
```
<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/skeweddistribution.PNG?raw=true" width="500" height="300" />

Note - Our target variable (SalePrice) is clearly skewed to the right. Most ML models don't do well with non-normally distributed data. We can apply a log(1+x) transformation to fix the skew. Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.

```python
# log(1+x) transformation
normalized = df.copy()
normalized["SalePrice"] = np.log1p(normalized["SalePrice"])

# Visualizing the distribution of our normalized target variable
sns.distplot(normalized['SalePrice'] , fit=norm);

# Calculate the fitted parameters used by the function
(mu, sigma) = norm.fit(normalized['SalePrice'])

# Plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.ticklabel_format(style='plain');

# Visualize the QQ-plot
fig = plt.figure()
res = stats.probplot(normalized['SalePrice'], plot=plt)
plt.show()
```
<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/normaldistribution.PNG?raw=true" width="500" height="300" />


