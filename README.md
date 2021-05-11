# Model 1 - Ty Pennington
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. This dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. With 79 explanatory variables describing (almost) every aspect of residential homes in Groveland MA, this data set challenges us to predict the final price of each home. However, what this data set does not account for is what I believe to be one of the most valuable determinants of price, "beauty lies in the eye of the beholder." In other words, the value of a house is often subjective - what one person values at 1,000,000 another may value at 875,000

Ty Pennington is an XGBoost predictive model trained to predict residential housing prices in Groveland MA. XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost has become a widely used and really popular tool among Kaggle competitors and Data Scientists, as it has been battle tested for production on large-scale problems. It is a highly flexible and versatile tool that can work through most regression, classification and ranking problems as well as user-built objective functions. 

Move that bus!

---

## Part 1 - Data Cleaning
Does my data contain outliers? What should I do about missing values? Is my data skewed? Why aren’t my dates formatted correctly? Data cleaning answers these questions and provides intuitive solutions. Arguably the most important step in the data science process, data cleaning can easily become deeply frustrating and time intensive. Recent studies suggest that data scientists spend anywhere from 50% to 80% of their time cleaning data rather than creating insights.

### Outliers
In plain english, an outlier is an observation that diverges from an overall pattern within a sample. Mathematically, an outlier is usually defined as an observation more than three standard deviations from the mean (although sometimes you'll see 2.5 or 2 as well). Most machine learning algorithms do not work well in the presence of outliers, as they are known to skew mean and standard deviation, reduce the effectiveness of statistical tests and decrease normality.

There is no precise way to define outliers in general because of the specifics of each data set. However, there are multiple methods that can be used to identify "potential" otliers. As the domain experts, we must interpret the observations and decide whether a value is an outlier or not. Said identification methods include Z-scores, Robust Z-scores, I.Q.R measurements and basic visualizations.

* `Univariate outliers` - can be found when we look at distribution of a single variable. Boxplots are commonly used to visualize and detect univariate outliers.
* `Multi-variate outliers` - are outliers in an n-dimensional space. In order to find them, you have to look at distributions in multi-dimensions.

```python
# Visualization of outliers
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

#### Note
I've decided to delete only two observations as they are blatant outliers (extremely large areas for very low prices). The training data probably contains additional outliers, however, removing all outliers may actually negatively impact our model if ever outliers were present in the test data. Instead of removing all outliers, I'll address skewed data in later sections and train our model to be robust on outliers.

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

#### Note
Our target variable (SalePrice) is clearly skewed to the right. Most ML models don't do well with non-normally distributed data. We can apply a log(1+x) transformation to fix the skew.

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
plt.xlabel('Percentage of Missing Columns', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Percent of Missing Values by Feature', fontsize=15);
```
<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/percentofmissingvalues.PNG?raw=true" width="500" height="500" />

```python
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
## Feature Engineering
The goal of feature engineering is simply to make our data better suited to the problem at hand. For a feature to be useful, it must have a relationship to the target that our model is able to learn. For example, linear models are only able to learn linear relationships. Therefore, when building a linear model, your goal is to transform input features so that their relationship to the target becomes linear. Common benefits of feature engineering include improved predictive performance, reduced computational needs and improved interpretability of results.

```python
# Some of the non-numeric predictors are stored as numbers --> convert them into strings 
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['OverallCond'] = df['OverallCond'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
```
#### Mutual Information
When presented with hundreds or thousands of description-less features, a new data set may often feel overwhelming. A great first step is to construct a ranking with a feature utility metric, a function measuring associations between a feature and the target. Said metric can be used to choose a smaller set of the most useful features to develop initially. Mutual information is a lot like correlation in that it measures a relationship between two quantities.

Mutual information describes relationships in terms of uncertainty. The mutual information (MI) between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. If you knew the value of a feature, how much more confident would you be about the target?

Mutual information is a great general-purpose metric and especially useful at the start of feature development: easy to use and interpret, computationally efficient, theoretically well-founded, resistant to overfitting and able to detect any kind of relationship. Once we've identified a set of features with some potential (see top 10 below), it's time to start developing them.

```python
# Creating our feature matrix (X) and target vector (y)
X = df.copy()
y = X.pop("SalePrice")

# Label encoding categorical columns
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
X['LotFrontage'] = X['LotFrontage'].astype(int)
discrete_features = X.dtypes == int
```
```python
# Creating a helper function to calculate our MI scores
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
```
```pyton
# Creating a helper function to plot our MI scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
```
<img src="https://github.com/jdonahue94/DonnyDoesDataScience1/blob/main/visualizations/mutualinformation.PNG?raw=true" width="600" height="400" />
