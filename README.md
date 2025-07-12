# 💸 Spends-Analytics-App: A Personal Finance Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL_HERE)

An interactive and mobile-friendly personal finance dashboard built with Streamlit. This app transforms your spending data into an engaging "Financial Feed," making it easy and intuitive to understand your financial habits.

![App Screenshot](https://raw.githubusercontent.com/streamlit/example-data/main/finance_app_demo.png)

## ✨ Core Features

* **Dynamic Financial Feed:** A chronological, social-media-style feed of your transactions, daily summaries, and weekly "Roll-ups."
* **Interactive Analytics:** Filter transactions by category with a single tap, and visualize daily spending with interactive charts.
* **Simulated LLM Chat:** Ask questions about your finances in plain English using a privacy-focused, simulated "Chat with Your Data" feature.
* **Predictive & Prescriptive Insights:** Get a 30-day spending forecast and receive "Smart Alerts" for recurring subscriptions and budget warnings.
* **Mobile-First Design:** A clean, responsive UI that works beautifully on both desktop and mobile devices.

## 🛠️ Tech Stack

* **Framework:** [Streamlit](https://streamlit.io/)
* **Data Handling:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Visualization:** [Plotly](https://plotly.com/)
* **Analytics:** [Scikit-learn](https://scikit-learn.org/)

## 🚀 How to Run Locally

Follow these steps to set up and run the project on your local machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/anuliketat/Spends-Analytics-App.git](https://github.com/anuliketat/Spends-Analytics-App.git)
cd YOUR_REPO_NAME
```

**2. Create a Virtual Environment (Recommended)**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
The `requirements.txt` file contains all necessary Python packages.
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit App**
```bash
streamlit run app.py
```
The application will open in your default web browser!

## ☁️ Deployment

This app is ready to be deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Simply link your GitHub repository and Streamlit will handle the rest.
