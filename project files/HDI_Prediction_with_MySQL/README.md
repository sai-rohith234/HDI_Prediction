# Human Development Index (HDI) Prediction System

A complete Machine Learning and Flask web application that estimates a country's Human Development Index (HDI) from health, education, and income indicators. The project is designed as an end-to-end learning example: data analysis, model training, visualization, serialization, and a responsive web interface.

## Features

- Cleans missing numeric data with column means.
- Performs exploratory analysis and creates a strip plot, two regression plots, and a correlation heatmap.
- Trains a scikit-learn `LinearRegression` pipeline using a reproducible train/test split.
- Reports the R² score and actual-versus-predicted HDI values.
- Saves the trained pipeline as `hdi_model.pkl`.
- Provides a Flask interface with server-side range validation and an accessible responsive design.

## Technologies Used

- Python 3.x, Flask, NumPy, Pandas
- Matplotlib, Seaborn, scikit-learn, Pickle
- HTML5 and CSS3

## Dataset

`Dataset/HumanDevelopmentIndex.csv` is a compact, educational dataset containing country-level development indicators. The columns used to train the model are:

- Life Expectancy
- Mean Years of Schooling
- Expected Years of Schooling
- Gross National Income per Capita

The target is `HDI`. The `Country` and `Development Group` columns are retained for identification and exploratory visualization.

## Folder Structure

```text
HDI_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── hdi_model.pkl
├── Dataset/
│   └── HumanDevelopmentIndex.csv
├── static/
│   ├── css/style.css
│   ├── images/
│   └── plots/
├── templates/
│   ├── home.html
│   ├── indexnew.html
│   └── result.html
└── notebooks/
    └── HDI_Analysis.ipynb
```

## Installation

```bash
cd HDI_Prediction
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## MySQL Database Setup

Install and start MySQL Server, then create the normalized database schema:

```bash
mysql -u root -p < schema.sql
```

The app reads its database settings from environment variables. The defaults are
`127.0.0.1:3306`, database `hdi_prediction_db`, user `root`, and an empty
password. Set your credentials before starting Flask:

```powershell
$env:HDI_DB_HOST = "127.0.0.1"
$env:HDI_DB_PORT = "3306"
$env:HDI_DB_NAME = "hdi_prediction_db"
$env:HDI_DB_USER = "root"
$env:HDI_DB_PASSWORD = "your_password"
```

`database.py` keeps reusable pooled connections and uses parameterized queries.
Each prediction is saved atomically across `users`, `country`,
`hdi_input_data`, `dataset`, `ml_model`, `hdi_prediction`, and `user_session`.
The existing prediction page has no login form, so its requests are associated
with a reusable **Anonymous Predictor** user and a unique browser session token.
If MySQL is temporarily unavailable, the ML prediction still renders and the
server logs the failed persistence attempt.

## How to Run

First generate the trained model and exploratory plots:

```bash
python train_model.py
```

Start the web application:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Select **Predict HDI**, enter all required indicators, and submit the form to view the estimate.

## Application Screens

The application includes a landing page, an indicator-entry page, and a prediction-result page. After running `train_model.py`, generated analysis images are available in `static/plots/`:

- `strip_plot_hdi.png`
- `life_expectancy_vs_hdi.png`
- `schooling_vs_hdi.png`
- `correlation_heatmap.png`

## Notes

This project is an educational predictive model. Its output is an estimate based on the bundled sample data and should not be interpreted as an official UNDP HDI calculation.
