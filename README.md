# 🧠 Digital Lifestyle & Mental Health Analytics

> *Transforming digital behavior data into actionable mental health insights.*

An end-to-end Data Analytics and Business Intelligence project that explores the relationship between digital lifestyle habits and mental health using the **Digital Lifestyle Benchmark Dataset**. The project follows a modern analytics workflow, incorporating data preprocessing, Medallion Architecture, Star Schema modeling, interactive Power BI dashboards, and machine learning to identify mental health risk factors and uncover meaningful behavioral patterns.

---

# 📋 Table of Contents

- Overview
- Project Team
- Dataset
- Project Objectives
- Tools & Technologies
- Project Architecture
- Data Warehouse
- Power BI Dashboard
- Machine Learning
- Key Insights
- Repository Structure
- Getting Started

---

# 📖 Overview

Digital technology has become an essential part of everyday life, influencing how people communicate, work, study, and spend their free time. While these technologies offer many benefits, excessive digital usage has also been associated with increased stress, anxiety, poor sleep, and other mental health concerns.

This project analyzes the relationship between digital lifestyle behaviors including screen time, social media usage, phone unlock frequency, notifications, sleep quality, and physical activity, and mental health indicators such as depression, anxiety, stress, and digital dependence.

Using Python for data processing, a Medallion Architecture for data transformation, a Star Schema for analytical modeling, and Power BI for visualization, the project transforms raw survey data into meaningful insights that support digital wellness awareness and data-driven decision making.

---

# 👥 Project Team

| Team Members |
|-------------|
| Malak Tamer |
| Khlood Ahmed |
| Salma Mohamed |
| Istabraq El Gendy |
| Hamza Bayoumi |
| Ahmed Hassan |

---

# 📊 Dataset

### Digital Lifestyle Benchmark Dataset

| Attribute | Details |
|-----------|---------|
| Dataset | Digital Lifestyle Benchmark Dataset |
| Source | HuggingFace |
| Records | 100,000 |
| Features | 24 |
| Format | CSV |

The dataset includes information across four main categories:

- 👤 Demographics
- 📱 Digital Lifestyle & Device Usage
- 🌙 Lifestyle & Wellness
- 🧠 Mental Health Indicators

---

# 🎯 Project Objectives

- Analyze the relationship between digital habits and mental health.
- Identify behavioral patterns associated with mental health risks.
- Build an efficient data warehouse using Star Schema.
- Develop interactive Power BI dashboards for business intelligence.
- Predict mental health risk using machine learning models.

---

# 🛠️ Tools & Technologies

| Category | Technologies |
|----------|--------------|
| Data Processing | Python, Pandas, NumPy |
| Data Cleaning | Python, Excel |
| Data Modeling | Star Schema |
| Data Warehouse | Medallion Architecture |
| Visualization | Power BI, DAX |
| Machine Learning | Scikit-learn, XGBoost |
| Dashboard Design | Figma |
| Development | Google Colab |

---

# 🏗️ Project Architecture

The project follows a simplified **Medallion Architecture** to organize and transform the data into analytics-ready datasets.

## 🥉 Bronze Layer

- Raw CSV dataset
- No transformations
- Landing zone for source data

---

## 🥈 Silver Layer

- Data validation
- Missing value & duplicate checks
- Business rule validation
- Standardized categories
- Feature engineering
- Creation of analytical columns

---

## 🥇 Gold Layer

- Business-ready datasets
- Star Schema implementation
- Optimized for Power BI reporting
- Machine learning ready

---

# 🏛️ Data Warehouse

A **Star Schema** was designed to improve analytical performance and simplify reporting.

### Fact Table

**Fact Digital Lifestyle**

Contains the project's key numerical measures including:

- Device Hours
- Depression Score
- Anxiety Score
- Digital Dependence Score
- Digital Wellness Score
- Age

### Dimension Tables

- 👤 Dim Person
- 📱 Dim Device
- 🌍 Dim Geography
- ❤️ Dim Health Profile

This structure enables efficient filtering, drill-down analysis, and optimized dashboard performance.

---

# 📊 Power BI Dashboard

The project includes an **11-page interactive dashboard** covering different aspects of digital lifestyle and mental health.

Dashboard Pages:

- Executive Summary
- Demographics
- Device Usage Analysis
- Social Media Analysis
- Sleep Analysis
- Mental Health Analysis
- Productivity Analysis
- High Risk Users
- Physical Activity Analysis
- Correlation Analysis
- Segmentation Analysis

The dashboard provides dynamic filtering, KPIs, drill-down capabilities, and interactive visualizations for deeper analysis.

---

# 🤖 Machine Learning

The project includes a machine learning module that predicts whether a user is at high mental health risk based on their digital behavior.

### Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

The final model was selected based on evaluation metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

# 📈 Key Insights

Some of the project's main findings include:

- Higher daily device usage is strongly associated with higher depression scores.
- Increased phone unlock frequency is linked to greater digital dependence.
- Better sleep duration and sleep quality are associated with lower mental health risk.
- Approximately **20%** of users were classified as high-risk.
- Digital dependence is one of the strongest indicators of mental health outcomes.

---

## 📌 Conclusion

This project demonstrates how data engineering, business intelligence, and machine learning can be combined to analyze real-world behavioral data. By transforming raw data into interactive visualizations and predictive insights, the project highlights the value of data-driven approaches in understanding digital lifestyle patterns and mental health.
