import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import re
import os
import io
import google.generativeai as genai
# New import for encryption. You may need to run: pip install cryptography
from cryptography.fernet import Fernet

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Custom CSS ---
def local_css(file_name):
    """Loads a local CSS file."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Optional: To apply custom styles, create a '{file_name}' file.")

local_css("style.css")


# --- ENCRYPTION/DECRYPTION FUNCTIONS for API KEY ---
# The following functions handle the secure storage of the Google AI API key.
# Two files will be created in your app's root directory:
# 1. .secret.key: This is your encryption key. DO NOT DELETE OR SHARE IT.
# 2. .env.encrypted: This file stores your encrypted API key.

def write_key():
    """Generates a key and save it into a file"""
    key = Fernet.generate_key()
    with open(".secret.key", "wb") as key_file:
        key_file.write(key)
    return key

def load_key():
    """Loads the key from the current directory named `'.secret.key'`"""
    try:
        return open(".secret.key", "rb").read()
    except FileNotFoundError:
        return write_key()

def encrypt_api_key(api_key, key):
    """Encrypts the API key and saves it."""
    f = Fernet(key)
    encrypted_key = f.encrypt(api_key.encode())
    with open(".env.encrypted", "wb") as encrypted_file:
        encrypted_file.write(encrypted_key)

def decrypt_api_key(key):
    """Decrypts the API key from the file."""
    try:
        f = Fernet(key)
        with open(".env.encrypted", "rb") as encrypted_file:
            encrypted_key = encrypted_file.read()
        decrypted_key = f.decrypt(encrypted_key).decode()
        return decrypted_key
    except FileNotFoundError:
        return None
    except Exception as e:
        # Handle cases where the key might be invalid or file corrupted
        st.error(f"Failed to decrypt API key. You may need to provide it again. Error: {e}")
        # Clean up corrupted files
        if os.path.exists(".env.encrypted"):
            os.remove(".env.encrypted")
        return None


# --- DATA PROCESSING FUNCTIONS ---

def get_category(merchant):
    """Categorizes transactions based on merchant name."""
    merchant = str(merchant).lower()
    categories = {
        "Food & Dining": ["zomato", "swiggy", "restaurant", "grocery", "bakery", "eats", "cafe", "dominos"],
        "Transport": ["uber", "ola", "metro", "fuel", "petrol", "rapido", "railway"],
        "Shopping": ["amazon", "flipkart", "myntra", "zara", "ajio", "shopping", "mart", "market"],
        "Bills & Utilities": ["airtel", "vodafone", "jio", "bsnl", "electricity", "bill", "credit card", "recharge", "gas"],
        "Entertainment": ["netflix", "spotify", "bookmyshow", "pvr", "inox", "movies", "prime video", "hotstar"],
        "Health & Wellness": ["pharmacy", "apollo", "netmeds", "clinic", "hospital", "medplus", "health"],
        "Transfers": ["upi", "p2a", "p2m", "transfer"],
    }
    for category, keywords in categories.items():
        if any(keyword in merchant for keyword in keywords):
            return category
    return "Other"

def parse_fallback_format(row_str):
    """Parses a single string row from the fallback format."""
    amount_match = re.search(r'INR\s+([\d,]+\.?\d*)', row_str, re.IGNORECASE)
    amount = float(amount_match.group(1).replace(',', '')) if amount_match else 0.0
    datetime_match = re.search(r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', row_str)
    if datetime_match:
        full_datetime = pd.to_datetime(datetime_match.group(1), format='%Y-%m-%d %H:%M:%S')
    else:
        date_match = re.search(r'(\d{2}-\d{2}-\d{2})', row_str)
        time_match = re.search(r'(\d{2}:\d{2}:\d{2})', row_str)
        date_str = date_match.group(1) if date_match else None
        time_str = time_match.group(1) if time_match else '00:00:00'
        if not date_str: return None
        full_datetime = pd.to_datetime(f"{date_str} {time_str}", format='%d-%m-%y %H:%M:%S')
    merchant_match = re.search(r'UPI/\w+/\w+/([\w\s.-]+)', row_str)
    if merchant_match:
        merchant = merchant_match.group(1).split('/')[0].strip()
    else:
        parts = row_str.split(' at ')
        if len(parts) > 1:
            merchant = parts[-1].split('.')[0].strip()
        else:
            merchant = 'Unknown'
    bank_match = re.search(r'(Axis Bank|HDFC Bank|ICICI Bank|SBI|State Bank of India)', row_str, re.IGNORECASE)
    bank = bank_match.group(1) if bank_match else 'Unknown'
    transaction_type = 'debit' if 'debit' in row_str.lower() else 'credit'
    if amount > 0 and transaction_type == 'debit':
        return {"Date": full_datetime, "Amount": amount, "Type": transaction_type, "Merchant": merchant, "Bank": bank}
    return None

@st.cache_data
def process_data(data_source):
    """Processes spending data from a file-like object."""
    try:
        df = pd.read_csv(data_source, header=0)
        df_copy = df.copy()
        df_copy.rename(columns={'datetime': 'Date', 'date': 'Date', 'amount': 'Amount', 'type': 'Type', 'debit/credit': 'Type', 'merchant': 'Merchant', 'recipient (merchant)': 'Merchant', 'recipient': 'Merchant', 'bank name': 'Bank', 'bank': 'Bank'}, inplace=True)
        required_cols = ['Date', 'Amount', 'Type', 'Merchant']
        if not all(col in df_copy.columns for col in required_cols):
            raise ValueError("Missing standard columns, attempting fallback parsing.")
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['Amount'] = pd.to_numeric(df_copy['Amount'], errors='coerce').fillna(0)
        df_copy['Type'] = df_copy['Type'].astype(str).str.lower()
        df_copy['Merchant'] = df_copy['Merchant'].astype(str)
        clean_df = df_copy[df_copy['Type'] == 'debit'].copy()
    except (ValueError, pd.errors.ParserError, KeyError):
        st.info("Standard CSV format not detected. Attempting fallback parsing...")
        data_source.seek(0)
        lines = data_source.readlines()
        if isinstance(lines[0], bytes): lines = [line.decode('utf-8', errors='ignore') for line in lines]
        extracted_data = [parse_fallback_format(line.strip()) for line in lines if line.strip()]
        extracted_data = [d for d in extracted_data if d is not None]
        if not extracted_data:
            st.error("Could not extract any valid transactions using either method.")
            return None
        clean_df = pd.DataFrame(extracted_data)
    if clean_df.empty:
        st.warning("No debit transactions found in the data.")
        return None
    clean_df['Category'] = clean_df['Merchant'].apply(get_category)
    return clean_df.sort_values(by="Date", ascending=False)

# --- LLM AND CHAT FUNCTIONS ---

def get_system_prompt():
    """Defines the persona and instructions for the LLM."""
    return """
    You are "Fin", a friendly, expert financial guide. Your goal is to help the user understand their spending habits and make smarter financial decisions.
    **Your Persona:**
    - **Identity:** A 30-year-old single male working in a competitive IT giant in Hyderabad. You understand the lifestyle, pressures, and financial opportunities of this demographic.
    - **Tone:** Knowledgeable, encouraging, and slightly informal. Like a wise friend or mentor. Never be judgmental.
    - **Method:** Provide clear, actionable insights based on the data provided. Use markdown, bolding, and lists to make your advice easy to digest.
    **Your Task:**
    1.  Analyze the user's spending data provided in the prompt.
    2.  Answer the user's specific question.
    3.  Provide one or two concise, actionable insights or a piece of advice related to their question.
    4.  **Proactively build the user's persona.** After answering, ask a simple, non-intrusive question to learn more about their financial goals.
    **Data Context:**
    - The user's transaction data will be provided below in a markdown format.
    - Today's date is {current_date}.
    """

@st.cache_data
def get_llm_response(api_key, query, _df, chat_history):
    """Generates a response from the LLM, with caching."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        data_summary = f"""
            Here is a summary of the user's recent financial data:
            **Basic Statistics:**
            ```{_df.describe().to_markdown()}```
            **Top 5 Spending Categories:**
            ```{_df.groupby('Category')['Amount'].sum().nlargest(5).to_markdown()}```
            **Top 5 Merchants by Spending:**
            ```{_df.groupby('Merchant')['Amount'].sum().nlargest(5).to_markdown()}```
            **Last 10 Transactions:**
            ```{_df.head(10).to_markdown(index=False)}```
        """
        system_prompt = get_system_prompt().format(current_date=datetime.now().strftime('%Y-%m-%d'))
        full_prompt = f"{system_prompt}\n\n{data_summary}\n\n**Previous Conversation:**\n{chat_history}\n\n**User's Question:**\n{query}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error. Please ensure your API key is correct and has access to the model. Error: {e}"

# --- Main App UI ---

st.title("💸 Your Financial Feed")
st.markdown("Load your spending data to generate an interactive dashboard.")

# --- Initialize Session State ---
if 'api_key' not in st.session_state: st.session_state.api_key = None
if 'filtered_category' not in st.session_state: st.session_state.filtered_category = None
if 'show_chat' not in st.session_state: st.session_state.show_chat = False
if "messages" not in st.session_state: st.session_state.messages = []

# --- Load and Handle API Key ---
encryption_key = load_key()
if st.session_state.api_key is None:
    st.session_state.api_key = decrypt_api_key(encryption_key)

# --- Data Source Selection ---
st.header("1. Choose Your Data Source")
source_option = st.radio("Select how to load your spending data:", ("Upload a CSV File", "Use Default Local File (Android)"), index=0, horizontal=True, label_visibility="collapsed")

df = None
data_source = None
if source_option == "Upload a CSV File":
    with st.expander("ℹ️ Show required CSV formats"):
        st.markdown("**1. Standard Format (Recommended):** A CSV with headers like `Date`, `Amount`, `Type`, `Merchant`.")
        st.dataframe(pd.DataFrame([{'Date': '2025-07-12 18:30:00', 'Amount': 250.75, 'Type': 'debit', 'Merchant': 'Zomato', 'Bank': 'Axis Bank'}, {'Date': '2025-07-11 11:00:00', 'Amount': 1200.00, 'Type': 'debit', 'Merchant': 'Amazon', 'Bank': 'HDFC Bank'}]), use_container_width=True)
        st.markdown("**2. Fallback Format:** A CSV with no headers, where each row is a single text block from an SMS.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file: data_source = uploaded_file
elif source_option == "Use Default Local File (Android)":
    local_filepath = "/storage/emulated/0/Documents/spends.csv"
    st.info(f"Attempting to load data from: `{local_filepath}`")
    if os.path.exists(local_filepath):
        try: data_source = open(local_filepath, 'rb')
        except Exception as e: st.error(f"Error opening local file: {e}")
    else: st.warning("Default file not found. Use the upload option.")

if data_source:
    df = process_data(data_source)
    if hasattr(data_source, 'close'): data_source.close()

# --- Dashboard Display (only if data is loaded) ---
if df is not None and not df.empty:
    st.success(f"Successfully loaded and processed {len(df)} transactions. Your dashboard is ready!")
    st.header("2. Your Analytical Dashboard")
    st.markdown("---")
    main_content = st.container()
    chat_placeholder = st.empty()

    with main_content:
        st.markdown("### Today's Snapshot")
        today_df = df[df['Date'].dt.date == datetime.now().date()]
        total_spent_today = today_df['Amount'].sum()
        daily_budget = 1500
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Spent Today", value=f"₹{total_spent_today:,.2f}")
            st.progress(min(total_spent_today / daily_budget, 1.0))
            st.caption(f"Daily Budget: ₹{daily_budget:,.2f}")
        with col2:
            if not today_df.empty:
                today_cat_dist = today_df.groupby('Category')['Amount'].sum().nlargest(3)
                fig = go.Figure(data=[go.Pie(labels=today_cat_dist.index, values=today_cat_dist.values, hole=.6, marker_colors=px.colors.sequential.RdBu)])
                fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=150, annotations=[dict(text='Top 3', x=0.5, y=0.5, font_size=16, showarrow=False)])
                st.plotly_chart(fig, use_container_width=True)
            else: st.write("No spending recorded yet today.")
        st.markdown("---")
        if st.session_state.filtered_category:
            col1, col2 = st.columns([3, 1])
            with col1: st.info(f"Showing transactions for: **#{st.session_state.filtered_category}**")
            with col2:
                if st.button("Clear Filter", use_container_width=True):
                    st.session_state.filtered_category = None
                    st.rerun()
            display_df = df[df['Category'] == st.session_state.filtered_category]
        else: display_df = df
        st.markdown("### Recent Activity")
        for index, row in display_df.head(20).iterrows():
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 4, 2])
                with col1: st.write(f"**{row['Date'].strftime('%b %d')}**")
                with col2:
                    st.markdown(f"**{row['Merchant']}**")
                    if st.button(row['Category'], key=f"cat_{index}", type="secondary"):
                        st.session_state.filtered_category = row['Category']
                        st.rerun()
                with col3: st.markdown(f"<div style='text-align: right; color: #D32F2F; font-weight: bold;'>- ₹{row['Amount']:.2f}</div>", unsafe_allow_html=True)

    # --- The LLM-Powered Chat ---
    if st.button("🤖 Chat with Fin, Your AI Guide"): st.session_state.show_chat = not st.session_state.show_chat
    if st.session_state.show_chat:
        with chat_placeholder.container(border=True):
            st.markdown("### 💬 Chat with Fin")
            if st.session_state.api_key is None:
                st.info("To enable the AI chat, please provide your Google AI API key. It will be encrypted and stored locally for future use.")
                new_api_key = st.text_input("Enter your Google AI API Key to begin:", type="password", key="api_key_input_new")
                if st.button("Save Key"):
                    if new_api_key:
                        encrypt_api_key(new_api_key, encryption_key)
                        st.session_state.api_key = new_api_key
                        st.success("API Key saved successfully! Refresh the page to start chatting.")
                        st.rerun()
                    else: st.warning("Please enter a valid API key.")
            else:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]): st.markdown(message["content"])
                if prompt := st.chat_input("Ask about your finances..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Fin is thinking..."):
                            history_for_llm = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                            response = get_llm_response(st.session_state.api_key, prompt, df, history_for_llm)
                            st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # --- Predictive and Prescriptive Analytics ---
    st.markdown("---")
    st.markdown("### 🔮 Your Financial Compass")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Spending Forecast")
            df_ts = df.set_index('Date').resample('D')['Amount'].sum().fillna(0).reset_index()
            df_ts['time'] = (df_ts['Date'] - df_ts['Date'].min()).dt.days
            if len(df_ts) > 30:
                model = LinearRegression().fit(df_ts[['time']], df_ts['Amount'])
                future_days = pd.DataFrame({'time': range(df_ts['time'].max() + 1, df_ts['time'].max() + 31)})
                future_days['Date'] = df_ts['Date'].min() + pd.to_timedelta(future_days['time'], unit='d')
                future_days['forecast'] = model.predict(future_days[['time']])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Amount'], mode='lines', name='Actual'))
                fig.add_trace(go.Scatter(x=future_days['Date'], y=future_days['forecast'], mode='lines', name='Forecast', line=dict(dash='dash')))
                fig.update_layout(title="Projected Spending (Next 30 Days)", xaxis_title=None, yaxis_title="Amount (₹)", margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else: st.write("Not enough data for a forecast (30+ days needed).")
    with col2:
        with st.container(border=True):
            st.markdown("#### Smart Alerts")
            recurring_payments = df[df['Category'] == 'Bills & Utilities']['Merchant'].value_counts()
            subscriptions = recurring_payments[recurring_payments >= 2]
            st.subheader("Subscription Slayer")
            if not subscriptions.empty: st.info(f"Recurring payments noticed to **{subscriptions.index[0]}**. Still using this?")
            else: st.success("No potential subscriptions found.")
            monthly_budget = 50000
            current_month_spending = df[df['Date'].dt.month == datetime.now().month]['Amount'].sum()
            st.subheader("Budget Shield")
            if current_month_spending > monthly_budget: st.error(f"Over budget by ₹{current_month_spending - monthly_budget:,.2f}!")
            elif current_month_spending > monthly_budget * 0.8: st.warning("You've spent over 80% of your monthly budget.")
            else: st.success("Spending is on track this month.")
else:
    st.info("Awaiting data to build your financial dashboard...")
