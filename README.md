# Machine Learning functionalities for a web service - Forecasting and Clustering

## Overview

This project implements back-end code for integrating **forecasting** and **clustering** functionalities into a web service. It enhances the end-user experience by enabling data-driven insights through intelligent automation and user-friendly data analysis tools. Additionally, the project includes **time series analytics** to aid users in interpreting their data visually and effectively.

The functionalities are designed to process data with minimal manual intervention, ensuring seamless usability for relatively non-technical end-users. 

This project highlights my ability to work with real-world data through preprocessing techniques such as optimizing memory usage, feature engineering, automation, and generating outputs using machine learning models.

---

## Features

### 1. Clustering
- **Methods Supported**: `KMeans`, `DBSCAN`, `Agglomerative`
- **Description**: Groups data observations based on user-selected features, allowing for insightful segmentation.
- **Outputs**:
  - A **CSV file** with cluster labels for each observation.
  - A **scatter plot** (for datasets with two axial variables).
<img width="300" alt="Capture d’écran 2024-11-21 à 15 52 07" src="https://github.com/user-attachments/assets/d9e004a1-de52-42e1-8ea2-209d1cdee285">

  - A **statistical summary** (`.txt` file) with metrics like cluster means and sizes.

### 2. Forecasting
- **Methods Supported**: `Holt-Winters (HW)`, `ARMA`, `ARIMA`, `SARIMA`
- **Description**: Predicts future trends in time series data based on historical patterns.
- **Outputs**:
  - A **CSV file** combining original and forecasted values.
  - A **graph** showcasing original and predicted values.
<img width="500" alt="Capture d’écran 2024-11-21 à 15 52 34" src="https://github.com/user-attachments/assets/9ba68df4-ddd7-4b57-836b-71833ddab432">


### 3. Time Series Analytics
- **Description**: Generates time series decomposition :
- **Outputs**: A **visualization** of the decomposition for easy interpretation.
<img width="500" alt="Capture d’écran 2024-11-21 à 15 51 55" src="https://github.com/user-attachments/assets/8244d7a7-19dd-45c6-aef3-f8a8a4a7f7a7">


---

## Tools & Technologies

- **Programming Language**: Python
- **Libraries**:
  - **Clustering**: `scikit-learn`
  - **Time Series Forecasting**: `statsmodels`
  - **Data Manipulation & Visualization**: `pandas`, `numpy`, `matplotlib`
- **Development Environment**: Visual Studio Code (VSCode)

---

## How to Use

Since the project is tailored to a specific dataset (in CSV format), which is confidential and cannot be shared publicly, it is not intended for direct cloning and usage without modifications. Below are the instructions to run the main functionalities.

### Clustering

Run the script `main_clustering.py` with the following arguments:
1. Path to the data (CSV file).
2. Name of the column containing the subject data.
3. List of feature column names.
4. Grouping function: `'mean'`, `'sum'`, or `'count'`.
5. Clustering method: `'kmeans'`, `'dbscan'`, or `'agglomerative'`.

**Example Command**:
```bash
python main_clustering.py ../datasets/dataset1.csv 'Country' "Age Number" 'mean' 'kmeans'
```

### Forecasting

Run the script main_forecasting.py with the following arguments:

1. Path to the data (CSV file).
2. Name of the column for years.
3. Name of the column for months.
4. Number of values to predict.
5. Forecasting method: 'HW', 'ARMA', 'ARIMA', or 'SARIMA'.
6. Name of the time series column (e.g., 'Number' or 'Price').
7. Test dataset length.
8. Season length (e.g., 12 for yearly seasonality).

**Example Command**:
```bash
python main_forecasting.py ../datasets/dataset1.csv 'Jahr' 'Monat' 7 'hw' 'Number' 5 12
```

### Time Series Analytics

Run the script main_ts_analytics.py with the following arguments:

1. Path to the data (CSV file).
2. Name of the column for years.
3. Name of the column for months.
4. Name of the time series column (e.g., 'Number' or 'Price').
5. Season length (e.g., 12 for yearly seasonality).

**Example Command**:
```bash
python main_ts_analytics.py ../datasets/dataset1.csv 'Jahr' 'Monat' 'Number' 12
```
