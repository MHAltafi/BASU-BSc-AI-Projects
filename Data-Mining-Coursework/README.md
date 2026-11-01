# Data Mining Coursework

This repository contains a collection of assignments and practical exercises completed during Fall 2023 as part of the *Introduction to Data Mining* course at Bu-Ali Sina University. The work explores different concepts and techniques in data mining and machine learning, including data preprocessing, classification, association rule mining, and anomaly detection, through hands-on implementations using Python and common data science libraries.

## Assignments

### 1. Anomaly Detection Using Scikit-learn and PyOD

An educational presentation exploring anomaly detection concepts and several approaches to identifying outliers. It introduces different types of anomalies, then demonstrates and visualizes multiple anomaly detection algorithms using synthetic datasets with artificially added outliers.

**Key topics:**
- Introduction to anomaly detection and its applications
- Types of anomalies: point, contextual, and collective
- Synthetic dataset generation using blobs, moons, and random data
- Adding artificial outliers to datasets
- Implementation and visualization of anomaly detection algorithms:
  - One-Class SVM
  - SGD One-Class SVM with Nystroem kernel approximation
  - Isolation Forest
  - Elliptic Envelope
  - Local Outlier Factor (LOF)
  - DBSCAN
- Decision boundary and anomaly detection visualization
- Exploring DBSCAN parameters using k-distance graphs and nearest neighbors
- Examining anomaly predictions and decision scores
- Introduction to PyOD and its anomaly detection capabilities
- References to documentation and learning resources

### 2. Association Rule Mining and Decision Tree Classification

**Dataset:** Congressional Voting Records (`house-votes-84`)

A comparative exploration of association rule mining and decision tree classification using the Congressional Voting Records dataset. The exercise examines how the two approaches can identify relationships between voting patterns and political affiliation.

**Key topics:**
- Exploratory data analysis and identification of missing values and duplicate records
- Conversion of categorical voting responses into numerical values
- One-hot encoding of political affiliations
- Class-based missing-value imputation using the most frequent values
- Frequent itemset generation using the Apriori algorithm
- Association rule extraction using support and confidence thresholds
- Analysis of voting patterns associated with Democrats and Republicans
- Decision tree classification with a limited tree depth
- Feature importance analysis and decision tree visualization
- Comparison of pattern discovery through association rules and classification through decision trees

### 3. Centroid-Based Classification from Scratch

**Dataset:** Iris

Implementation of a centroid-based classification algorithm from scratch using NumPy, following the Scikit-learn estimator interface (`fit`, `predict`, and `score`). The classifier assigns each sample to the class with the nearest centroid based on Euclidean distance.

**Key topics:**
- Exploratory data analysis and pairplot visualization
- Class centroid calculation using training data
- Euclidean distance-based classification
- Scikit-learn-style model implementation
- Training and test data separation
- Experiments with different feature combinations:
  - Sepal length and petal length
  - Sepal length and sepal width
  - Petal length and petal width
  - All four features
- Classification accuracy evaluation on test data
- Decision-region visualization for two-feature combinations

### 4. Obesity Classification Using Decision Trees and Random Forests

**Dataset:** Obesity Levels (`ObesityDataSet`)

Classification of obesity levels using Decision Tree and Random Forest algorithms. The project explores the dataset through exploratory data analysis and preprocessing, then compares the performance of both classification models using a stratified train/test split.

**Key topics:**
- Exploratory data analysis and statistical summaries
- Numerical feature visualization using boxplots
- Missing-value and duplicate-record detection and removal
- Binary and ordinal categorical feature encoding
- One-hot encoding of remaining categorical features
- Stratified training and test data splitting
- Decision Tree classification
- Random Forest classification
- Model evaluation using classification reports
- Confusion matrix visualization for model comparison

## Technologies and Libraries

- Jupyter Notebook
- Matplotlib
- Mlxtend
- NumPy
- Pandas
- PyOD
- Python
- Scikit-learn
- Seaborn

## Repository Structure

```text
Data-Mining-Coursework/
│
├── .gitignore
├── README.md
│
├── Anomaly Detection/
│   ├── Anomaly Detection Using Sklearn.ipynb
│   └── figs/
│
├── Association Rule Mining and Decision Trees/
│   ├── Association Rule Mining and Decision Trees.ipynb
│   ├── house-votes-84.csv        # ignored (dataset)
│   └── house-votes-84.names      # ignored (dataset)
│
├── Centroid-Based Classification from Scratch/
│   ├── Centroid-Based Classification from Scratch.ipynb
│   └── Iris_analysis.xlsx        # ignored (dataset)
│
└── Obesity Classification Using Decision Trees and Random Forests/
    ├── Obesity Classification with DT and RF.ipynb
    └── ObesityDataSet.csv        # ignored (dataset)
```

## Purpose

The purpose of this coursework is to gain practical experience with data mining and machine learning techniques through the implementation and analysis of different algorithms. The assignments cover data preprocessing, pattern discovery, classification, and anomaly detection, combining theoretical concepts with practical experiments.

They also provide an opportunity to explore different algorithms, visualize their behavior, compare their results, and understand how machine learning methods can be applied to different datasets and problems.