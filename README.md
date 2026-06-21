\# Customer Analytics and Sales Prediction System



A complete end-to-end Machine Learning project for predicting customer churn in banking.



Built with Python, Scikit-learn, and Streamlit.



\## Project Overview



This project builds a complete Customer Analytics system that:

\- Analyzes 10,000 real bank customers

\- Predicts which customers will leave (churn)

\- Provides confidence scores and business recommendations

\- Deploys as an interactive web application



\## Tech Stack



\- Python 3.11+

\- Pandas and NumPy for data manipulation

\- Scikit-learn for machine learning

\- Matplotlib and Seaborn for visualization

\- Streamlit for web application

\- Joblib for model persistence

\- PyYAML for configuration management



\## Machine Learning Models



Five models were trained and compared:



\- Logistic Regression: Simple baseline model

\- Random Forest: Ensemble of decision trees

\- Gradient Boosting: Sequential boosting (WINNER)

\- Support Vector Machine: Optimal boundary classifier

\- K-Nearest Neighbors: Similarity based classifier



Best Model: Gradient Boosting



\## Key Results



\- Dataset: 10,000 bank customers

\- Features: 22 engineered features

\- Best Model: Gradient Boosting

\- Churn Rate in Dataset: 20.4%



\## Project Structure



\- config/ — All project settings

\- data/ — Raw and processed datasets

\- src/ — All Python source code

\- app/ — Streamlit web application

\- models/ — Saved trained models

\- reports/ — Generated charts and figures

\- logs/ — Application logs



\## Installation



Step 1: Clone the repository



git clone https://github.com/yourusername/customer\_analytics.git

cd customer\_analytics



Step 2: Create virtual environment



python -m venv venv

venv\\Scripts\\activate



Step 3: Install dependencies



pip install -r requirements.txt



Step 4: Download dataset



kaggle datasets download -d shrutimechlearn/churn-modelling -p data/raw --unzip



Step 5: Run the pipeline



python main.py



Step 6: Launch web app



streamlit run app/streamlit\_app.py



\## Features



\- Professional folder structure

\- Modular Python code

\- Type hints throughout

\- Comprehensive logging

\- Exception handling

\- Configuration management

\- Interactive web dashboard

\- CSV export functionality

\- Business insights and recommendations



\## Business Value



This system helps banks:

\- Identify at-risk customers before they leave

\- Take proactive retention actions

\- Prioritize customers by risk level

\- Save revenue through targeted retention



\## Author



Your Name

Final Year B.Sc Computer Science Student



\## License



MIT License

