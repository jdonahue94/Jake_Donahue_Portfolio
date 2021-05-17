# Model 1 - Ty Pennington
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. This dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. With 79 explanatory variables describing (almost) every aspect of residential homes in Groveland MA, this data set challenges us to predict the final price of each home. However, what this data set does not account for is what I believe to be one of the most valuable determinants of price, "beauty lies in the eye of the beholder." In other words, the value of a house is often subjective - what one person values at 1,000,000 another may value at 875,000

Ty Pennington is an XGBoost predictive model trained to predict residential housing prices in Groveland MA. XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost has become a widely used and really popular tool among Kaggle competitors and Data Scientists, as it has been battle tested for production on large-scale problems. It is a highly flexible and versatile tool that can work through most regression, classification and ranking problems as well as user-built objective functions. 

Move that bus!

---

## Part 1 - Data Cleaning
Does my data contain outliers? What should I do about missing values? Is my data skewed? Why aren’t my dates formatted correctly? Data cleaning answers these questions and provides intuitive solutions. Arguably the most important step in the data science process, data cleaning can easily become deeply frustrating and time intensive. Recent studies suggest that data scientists spend anywhere from 50% to 80% of their time cleaning data rather than creating insights.

### Handling Missing Values
Python libraries represent missing numbers as NaN which is short for "not a number". Most libraries (including scikit-learn) will give you an error if you try to build a model using data with missing values. In general, one can either drop columns with missing values or impute missing values. Dropping columns entirely can be useful when most values in a column are missing. Imputation fills in the missing value with some number. The imputed value won't be exactly right, however, it helps to produce more accurate predictive models.

```python
# Calculating and visualizing the percent of missing values per feature
sns.set_style("darkgrid")
missing = df.isnull().sum()
missing = missing[missing > 0]
missing = (missing / len(df)) * 100
missing.sort_values(ascending=True, inplace=True)
sns.barplot(x=missing.tail(15), y=missing.tail(15).index)
plt.xlabel("Percentage of Missing Columns", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Percent of Missing Values by Feature", fontsize=15);
```
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
# Slightly more intuitive fills
dataset.drop("LotFrontage", axis=1, inplace=True)
neighborhoodfrontage = train.groupby("Neighborhood")[["LotFrontage"]].median()#.rename({"Median": "LotFrontage"}, axis=1)
dataset = dataset.merge(neighborhoodfrontage, left_on="Neighborhood", right_index=True, how="left")
```
### Baseline Model
Establishing a baseline score will allow us to monitor our model's predictive performance following additional feature engineering. There are three metrics that are commonly used to evaluate the predictive performance of a regression model.

* `Mean Squared Error (MSE)` - The MSE is calculated as the mean or average of the squared differences between predicted and expected target values. Squaring has the effect of inflating or magnifying large errors. That is, the larger the difference between the predicted and expected values, the larger the resulting squared positive error. This has the effect of “punishing” models more for larger errors. The calculation of MSE is considered fairly sensitive to outliers.

* `Root Mean Squared Error (RMSE)` - Mathematically speaking, the RMSE is the square root of the MSE. It is a good idea to first establish a baseline RMSE using a naive predictive model. A model that achieves an RMSE better than the RMSE for the naive model has skill. Again, it is important to note that RMSE, similar to MSE, punishes models for larger errors and is sensitive to outliers.

* `Root Mean Squared Logarithmic Error (RMSLE)` - Is an extension on RMSE that is commonly used when a regression model is trained on large actual and predicted values. For example, when predicting residential housing prices, values can range from $100,000 to $1,000,000. In the case of RMSLE, we take the log of our model's predictions and actual values so that large errors and outliers are scaled down to limit their effect.

```python
# MSE Scoring
def score_dataset(model, df):
  
  # initiate X and y
  X = df.copy()
  y = X.pop("SalePrice")
  
  # preprocessing of X and y
  X = pd.get_dummies(X).reset_index(drop=True)
  log_y = np.log(y)
  
  # cross validation
  scores = cross_val_score(model, X, log_y, cv=5, scoring="neg_mean_squared_error")
  score = -1 * scores.mean()
  score = np.sqrt(score)
  return score
```
```python
# Calling the above function
boost = xgb.XGBRegressor()
baseline = score_dataset(boost, train)
print(f"Baseline RMSLE score = {baseline}")
```

### Outliers
In plain english, an outlier is an observation that diverges from an overall pattern within a sample. Mathematically, an outlier is usually defined as an observation more than three standard deviations from the mean (although sometimes you'll see 2.5 or 2 as well). Most machine learning algorithms do not work well in the presence of outliers, as they are known to skew mean and standard deviation, reduce the effectiveness of statistical tests and decrease normality.

There is no precise way to define outliers in general because of the specifics of each data set. However, there are multiple methods that can be used to identify "potential" otliers. As the domain experts, we must interpret the observations and decide whether a value is an outlier or not. Said identification methods include Z-scores, Robust Z-scores, I.Q.R measurements and basic visualizations.

* `Univariate outliers` - can be found when we look at distribution of a single variable. Boxplots are commonly used to visualize and detect univariate outliers.
* `Multi-variate outliers` - are outliers in an n-dimensional space. In order to find them, you have to look at distributions in multi-dimensions.

```python
# Multi-variate analysis saleprice/grlivarea
sns.set_style('darkgrid')
fig = sns.scatterplot(data=df, x=df['GrLivArea'], y=df['SalePrice'])
fig.set(xlabel='Living Area', ylabel='Sale Price', title='Living Area vs Sale Price (w/outliers)');
```
<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/outliarzzz.PNG?raw=true" width="500" height="300" />

```python
# intuitively deleting outliers (bottom right corner)
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)

# Visualization w/o outliers
sns.set_style('darkgrid')
fig = sns.scatterplot(data=df, x=df['GrLivArea'], y=df['SalePrice'])
fig.set(xlabel='Living Area', ylabel='Sale Price', title='Living Area vs Sale Price (w/o outliers)');
```
<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/nooutliarz.PNG?raw=true" width="500" height="300" />

Note - I've decided to delete only two observations as they are blatant outliers (extremely large areas for very low prices). The training data probably contains additional outliers, however, removing all outliers may actually negatively impact our model if ever outliers were present in the test data. Instead of removing all outliers, I'll address skewed data in later sections and train our model to be robust on outliers.

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

### Feature Engineering
Feature engineering is the process of extracting features or input variables from raw data which can be applied to our model to improve predictive performance and interpretability of results, while reducing computational needs. The process involves a combination of statistical analysis, domain knowledge and creativity.







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


