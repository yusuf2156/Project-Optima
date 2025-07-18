# **Project Optima: An Interactive Portfolio Optimization Engine** 

### **Live Demo: [Link to your deployed Streamlit App]** <--- We will add this link in the next phase!

## 🚀 Overview
Project Optima is an interactive financial analysis tool that built with Python which implements Modern Portfolio Theory(MPT).
To better understand the app,firstly, it needed to be understand how theory works. In a simple analogy,
if a person has 100$ and he/she wants to allocate this money between a bundle of stocks, person need to find a optimal 
balance between stocks.This helps to find optimal allocation between stocks. For example, 50 dollars for Google, 30 dollars for Amazon,
20 dollars for Apple.According to Modern Portfolio Theory, there is a mathematical way to do this which is sharpe ratio.
A ratio between return of stocks and risk(volatility). To sum up, It fetches real-world stock data, performs the necessary quantitative analysis,
## ✨ Key Features
* **Dynamic Data Fetching:** Fetches historical stock price data from Yahoo Finance using the `yfinance` library.
* **Quantitative Analysis:** Calculates annualized returns, volatility (risk), and the covariance matrix for a given set of assets using `NumPy` and `Pandas`.
* **Numerical Optimization:** Uses the `SciPy` library to perform a constrained numerical optimization to find the portfolio with the maximum Sharpe Ratio.
* **Monte Carlo Simulation:** Runs a Monte Carlo simulation to generate thousands of random portfolio weightings, visualizing the entire set of investment possibilities.
* **Interactive Dashboard:** A user-friendly web interface built with `Streamlit`

## 🧪 Examples & Results

###  Test Case 1: Modern Tech Giants

A high-growth, high-volatility portfolio.

**Tickers:** META, AAPL, AMZN, NFLX, GOOGL

**Start Date:** 2021-01-01

**End Date:** 2023-12-31

**Image of Experiments:** 
![Optimal Asset Allocation and Important numbers](https://i.imgur.com/RD5ZIwy.png)
![Efficient Frontier](https://i.imgur.com/LDf5Rm4.png)


### Test Case 2: Classic Diversified Portfolio

Demonstrating risk reduction through diversification across different sectors.

**Tickers:** JPM, JNJ, XOM, WMT, MSFT

**Start Date:** 2021-01-01

**End Date:** 2023-12-31

**Image of Experiment:** 
![Optimal Asset Allocation and Important numbers](https://i.imgur.com/LgXCKUt.png)
![Efficient Frontier](https://i.imgur.com/CXgHZjd.png)



### Test Case 3: Renewable Energy Thematic Portfolio

A high-risk, high-reward portfolio focused on a specific growth industry.

**Tickers:** ENPH, SEDG, FSLR, NEE, TSLA

**Start Date:** 2020-01-01

**End Date:** 2022-12-31

**Image of Experiment:** 
![Optimal Asset Allocation and Important numbers](https://i.imgur.com/uWRXX1B.png)
![Efficient Frontier](https://i.imgur.com/FjoGcpP.png)


### Test Case 4: Defensive & Dividends Portfolio

A conservative portfolio of stable, low-volatility companies.

**Tickers:** PG, KO, WM, DUK, PEP

**Start Date:** 2019-01-01

**End Date:** 2023-12-31

**Image of Experiment:** 
![Optimal Asset Allocation and Important numbers](https://i.imgur.com/okwDiL4.png)
![Efficient Frontier](https://i.imgur.com/0DH64yy.png)

### Test Case 5: Single Stock vs. The Market

A two-asset case demonstrating a core MPT(Modern Portfolio Theory) concept.

**Tickers:** NVDA, SPY

(NVIDIA vs. the S&P 500 ETF)

**Start Date:** 2022-01-01

**End Date:** 2024-06-30

**Image of Experiment:**
![Optimal Asset Allocation and Important numbers](https://i.imgur.com/Nb9Dzz8.png)
![Efficient Frontier](https://i.imgur.com/bOQ1THq.png)





## 🛠️ Technology Stack
| **Category**      | **Tools**                        |
|-------------------|----------------------------------|
| Language          | Python 3.9+                      |
| Data              | `pandas`, `numpy`, `yfinance`    |
| Optimization      | `scipy`                          |
| Visualization     | `plotly`, `streamlit`            |
| Deployment        | Streamlit Community Cloud        |

## 📦 Installation & Usage
To run this application locally, please follow these steps:
### 1. Clone the repository:
```
git clone https://github.com/your-username/project-optima.git
cd project-optima
```
### 2. Create and activate a virtual environment (recommended):
```python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
3. ### Install the required dependencies:
```
pip install -r requirements.txt
```
4. ### Run the Streamlit application:
```
streamlit run app.py
```
The application will open in your web browser.







