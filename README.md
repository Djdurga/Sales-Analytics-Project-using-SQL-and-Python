# 🧮 Sales Analytics Project using SQL and Python

This project demonstrates how to perform **data analysis** on the [Online Retail II Dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) using **MySQL and Python**. It walks through the full data pipeline: importing data, setting up SQL schema, connecting SQL to Python, performing exploratory data analysis (EDA), and customer segmentation.

---

## 📁 Project Structure

```
Sales_Analytics/
│
├── data/
│   └── online_retail_II.csv           # Raw dataset
├── scripts/
│   ├── sql_schema.sql                 # SQL schema and insert queries
│   └── eda_analysis.py                # Python code for EDA and clustering
├── README.md                          # Project documentation
└── customer_segments.csv              # Output from clustering analysis
```

---

## ⚙️ Prerequisites

- Python 3.x
- MySQL Server
- MySQL Workbench or CLI
- Python Libraries:
  - `pandas`
  - `sqlalchemy`
  - `pymysql`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install them using:

```bash
pip install pandas sqlalchemy pymysql matplotlib seaborn scikit-learn
```

---

## 🧵 Step-by-Step Guide

### 1️⃣ Load Dataset into Python

```python
import pandas as pd

df = pd.read_csv("online_retail_II.csv")
df.dropna(subset=["Customer ID"], inplace=True)
df['Customer ID'] = df['Customer ID'].astype(int)
df['Total'] = df['Quantity'] * df['Price']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
```

---

### 2️⃣ Create SQL Tables in MySQL

Run the following in MySQL Workbench:

```sql
CREATE DATABASE Sales_Analytics;
USE Sales_Analytics;

CREATE TABLE RawRetail (
    Invoice VARCHAR(20),
    StockCode VARCHAR(20),
    Description TEXT,
    Quantity INT,
    InvoiceDate DATETIME,
    Price DECIMAL(10, 2),
    CustomerID INT,
    Country VARCHAR(100)
);
```

---

### 3️⃣ Connect Python to MySQL

#### 🔐 Format:

```python
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://username:password@localhost:3306/Sales_Analytics")
```

Example:

```python
engine = create_engine("mysql+pymysql://root:Durga76@localhost:3306/Sales_Analytics")
```

> ⚠️ Replace `root`, `Durga76`, `localhost`, and `Sales_Analytics` with your actual MySQL username, password, host, and database name.

---

### 4️⃣ Upload DataFrame to SQL

```python
df.to_sql('RawRetail', con=engine, if_exists='replace', index=False)
```

---

### 5️⃣ Analyze the Data from SQL in Python

```python
query = """
    SELECT InvoiceDate, StockCode, Description, Quantity, Price, CustomerID, Country, 
           (Quantity * Price) AS Total 
    FROM RawRetail;
"""
df = pd.read_sql(query, con=engine)

# Monthly Sales Trend
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['Total'].sum()
monthly_sales.plot(kind='line', title='Monthly Sales Trend')
```

---

### 6️⃣ RFM Segmentation (Customer Clustering)

```python
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
    'Invoice': 'nunique',
    'Total': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm)
```

---

## 📊 Key Outputs

- Monthly sales trends
- Top-selling products
- Country-wise sales breakdown
- Customer clusters based on RFM segmentation

---

## 💡 Highlights

- 🔗 SQL-to-Python pipeline using SQLAlchemy
- 📦 Clean, scalable project structure
- 📈 Visual analytics using Matplotlib & Seaborn
- 🔍 Insightful segmentation using machine learning

---

## 🤝 Credits

- Dataset: [Kaggle - Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- Author: Durga Rani

---

## 🛠️ Future Improvements

- Add interactive dashboard using Streamlit or Dash
- Schedule regular database updates
- Deploy model predictions for marketing use

---

## 📌 License

This project is under the MIT License.
