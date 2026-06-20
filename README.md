# Iris Dataset Exploratory Data Analysis (EDA)

## Project Overview

This project performs **Exploratory Data Analysis (EDA)** on the famous **Iris Flower Dataset**, one of the most widely used datasets in data science and machine learning. The analysis aims to uncover patterns, trends, correlations, and statistical insights within the dataset through descriptive statistics and data visualization techniques.

The primary objective is to understand the dataset structure, identify relationships between features, detect potential outliers, and evaluate feature importance for future machine learning applications.

---

## Dataset Information

The Iris dataset contains measurements of iris flowers belonging to three different species:

* **Setosa**
* **Versicolor**
* **Virginica**

### Features

| Feature      | Description                      |
| ------------ | -------------------------------- |
| Sepal Length | Length of the sepal (cm)         |
| Sepal Width  | Width of the sepal (cm)          |
| Petal Length | Length of the petal (cm)         |
| Petal Width  | Width of the petal (cm)          |
| Species      | Flower species (target variable) |

### Dataset Size

* Total Records: **150**
* Features: **4 Numerical Features**
* Target Classes: **3 Species**
* Missing Values: **None**

---

## Tools and Libraries Used

| Tool / Library   | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Primary programming language             |
| Pandas           | Data loading, manipulation, and analysis |
| Matplotlib       | Basic plotting and visualization         |
| Seaborn          | Statistical data visualization           |
| Jupyter Notebook | Interactive analysis environment         |

---

## Project Workflow

### Step 1: Data Loading and Inspection

The dataset was loaded using Pandas and inspected to verify data quality and structure.

#### Tasks Performed

* Loaded dataset using `pandas.read_csv()`
* Displayed sample records
* Checked dataset dimensions
* Verified column names
* Identified missing values

#### Functions Used

```python
iris.head()
iris.shape
iris.isnull().sum()
```

#### Findings

* Dataset contains **150 rows** and **5 columns**
* No missing values were found
* Data is clean and ready for analysis

---

### Step 2: Summary Statistics

Descriptive statistics were calculated to understand feature distributions and central tendencies.

#### Functions Used

```python
iris.describe()
iris.groupby("species").describe()
```

#### Findings

* **Setosa** has the smallest petals with an average petal length of **1.46 cm**
* **Virginica** has the largest petals with an average petal length of **5.55 cm**
* Significant variation exists among species, particularly in petal measurements

---

### Step 3: Data Visualization

Visualizations were created to better understand feature distributions and relationships.

---

#### A. Histograms

Purpose:

* Analyze feature distributions
* Detect skewness and multimodal patterns

```python
iris.hist(figsize=(10,8))
```

##### Observations

* **Sepal Width** follows an approximately normal distribution
* **Petal Length** exhibits a bimodal distribution due to species differences

---

#### B. Boxplots

Purpose:

* Compare feature distributions across species
* Detect potential outliers

```python
sns.boxplot(data=iris, x="species", y="sepal_length")
```

##### Observations

* Setosa generally has shorter but wider sepals
* Virginica has the longest petals
* Outliers are visible in certain measurements

---

#### C. Scatter Plot

Purpose:

* Visualize relationships between features
* Identify natural clustering patterns

```python
sns.scatterplot(
    data=iris,
    x="petal_length",
    y="petal_width",
    hue="species"
)
```

##### Observations

* Setosa forms a clearly separated cluster
* Versicolor and Virginica show slight overlap
* Petal measurements provide strong discriminatory power

---

#### D. Pair Plot

Purpose:

* Examine pairwise relationships among all numerical features

```python
sns.pairplot(iris, hue="species")
```

##### Observations

* Petal-related features demonstrate strong species separation
* Clear clustering patterns emerge among species

---

#### E. Correlation Heatmap

Purpose:

* Measure linear relationships between features

```python
sns.heatmap(iris.corr(), annot=True)
```

##### Observations

* Petal Length and Petal Width show a very strong positive correlation (**r = 0.96**)
* Sepal Width exhibits weak negative correlations with several features

---

### Step 4: Outlier Detection

Boxplots were used to identify unusual observations.

#### Code

```python
sns.boxplot(data=iris["sepal_width"])
```

#### Findings

* Most Sepal Width values fall within the expected range of **2.0–4.0 cm**
* A few observations appear outside the normal range and may require verification

---

## Key Findings

### Species Differentiation

* **Setosa** is highly distinct from the other species due to its small petals and wider sepals.
* **Versicolor** and **Virginica** overlap slightly but can still be differentiated using petal measurements.

### Feature Importance

* **Petal Length** and **Petal Width** are the most informative features for species classification.
* Sepal measurements contribute less to species separation.

### Correlation Analysis

* Strong correlation exists between petal dimensions (**r > 0.90**).
* High correlation suggests potential feature redundancy in predictive modeling.

### Outlier Analysis

* A small number of outliers were detected in the Sepal Width feature.
* These observations should be reviewed before building machine learning models.

---

## Conclusion

The exploratory analysis reveals that petal-related features are the strongest indicators for distinguishing Iris species. Visualizations demonstrate clear clustering patterns, particularly for Setosa, while correlation analysis highlights strong relationships between petal measurements.

The dataset is clean, well-structured, and suitable for machine learning applications such as classification, clustering, and predictive modeling.

---

## Future Work

Possible next steps include:

* Feature engineering
* Data preprocessing and scaling
* Machine Learning Classification Models

  * Logistic Regression
  * Decision Trees
  * Random Forest
  * Support Vector Machines (SVM)
  * K-Nearest Neighbors (KNN)
* Model evaluation and comparison
* Hyperparameter tuning

---

## Author

**WAjiha Waqar**

Exploratory Data Analysis of the Iris Dataset using Python, Pandas, Matplotlib, and Seaborn.
