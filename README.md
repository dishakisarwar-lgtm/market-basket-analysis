
# Market Basket Analysis

## Project Overview
This project analyzes shopping transactions to identify products that are frequently purchased together.

## Technologies Used
- Python
- Pandas
- MLxtend
- Matplotlib
- Streamlit

## Features
- Transaction data analysis
- Frequent itemset mining using the Apriori algorithm
- Association rules generation
- Support, confidence, and lift calculation
- Interactive dashboard with sliders
- CSV download of association rules

## Project Structure
- `app.py` - Interactive dashboard
- `market_basket.py` - Python analysis script
- `transactions.csv` - Sample transaction dataset
- `frequent_itemsets.png` - Frequent itemsets visualization
- `association_rules.png` - Association rules visualization

## Installation
Install the required packages:

```bash
python -m pip install pandas mlxtend matplotlib streamlit
```

## Run the Dashboard

```bash
python -m streamlit run app.py
```

## Objective
To discover product associations and identify potential cross-selling and product recommendation opportunities.

## Dataset
This project currently uses sample shopping transaction data for demonstration.
