import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import re
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Custom CSS ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"The '{file_name}' file was not found. Please ensure it's in the same directory as 'app.py'.")

local_css("style.css")

# --- DATA PROCESSING FUNCTIONS ---

def get_category(merchant):
    """Categorizes transactions based on merchant name."""
    merchant = str(merchant).lower()
    categories = {
        "Food": ["zomato", "swiggy", "restaurant", "grocery", "bakery"],
        "Transport": ["uber", "ola", "metro", "fuel", "petrol"],
        "Shopping": ["amazon", "flipkart", "myntra", "zara", "ajio", "shopping"],
        "Bills": ["airtel", "vodafone", "jio", "bsnl", "electricity", "bill", "credit card"],
        "Entertainment": ["netflix", "spotify", "bookmyshow", "pvr", "inox", "movies"],
        "Health": ["pharmacy", "apollo", "netmeds", "clinic", "hospital"],
    }
    for category, keywords in categories.items():
        if any(keyword in merchant for keyword in keywords):
            return category
    return "Other"

@st.cache_data
def process_local_spends_file(filepath):
    """
    Processes the specific 3-column SMS export format from the local Android path.
    Extracts Date, Time, Amount, and Merchant.
    """
    if not os.path.exists(filepath):
        st.warning(f"Default file not found at '{filepath}'. Please use the upload option or place the file in the correct path.")
        return None

    try:
        df = pd.read_csv(filepath, header=None, names=['datetime_col', 'col_b', 'col_c'])
        
        extracted_data = []
        for _, row in df.iterrows():
            # --- Extract Amount ---
            amount_match = re.search(r'INR\s+([\d,]+\.?\d*)', str(row['col_b']))
            amount = float(amount_match.group(1).replace(',', '')) if amount_match else 0.0

            if amount == 0.0: continue # Skip if no amount is debited

            # --- Extract Date & Time ---
            date_match = re.search(r'(\d{2}-\d{2}-\d{2})', str(row['col_b']))
            time_match = re.search(r'(\d{2}:\d{2}:\d{2})', str(row['col_c']))
            
            date_str = date_match.group(1) if date_match else None
            time_str = time_match.group(1) if time_match else '00:00:00'
            
            if date_str is None: continue # Skip if no date found
            
            full_datetime = pd.to_datetime(f"{date_str} {time_str}", format='%d-%m-%y %H:%M:%S')

            # --- Extract Merchant ---
            merchant_parts = str(row['col_c']).split('/')
            merchant = merchant_parts[-1].strip() if merchant_parts else 'Unknown'

            extracted_data.append({
                "Date": full_datetime,
                "Merchant": merchant,
                "Amount": amount,
            })
        
        if not extracted_data:
            st.error("Could not extract any valid transactions from the local file. Please check the file format.")
            return None

        clean_df = pd.DataFrame(extracted_data)
        clean_df['Category'] = clean_df['Merchant'].apply(get_category)
        return clean_df.sort_values(by="Date", ascending=False)

    except Exception as e:
        st.error(f"An error occurred while processing the local file: {e}")
        return None

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Processes the user-uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        # Standardize column names
        df.rename(columns={
            'DateTime': 'Date', 'datetime': 'Date', 'date': 'Date',
            'Recipient (Merchant)': 'Merchant', 'recipient': 'Merchant', 'merchant': 'Merchant',
            'Amount': 'Amount', 'amount': 'Amount',
            'Debit/Credit': 'Type', 'type': 'Type'
        }, inplace=True)

        if 'Date' not in df.columns or 'Amount' not in df.columns or 'Merchant' not in df.columns:
            st.error("Uploaded file must contain 'Date', 'Amount', and 'Merchant' columns.")
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        # Filter for debits if type column exists
        if 'Type' in df.columns:
            df = df[df['Type'].str.lower() == 'debit']
        
        df['Category'] = df['Merchant'].apply(get_category)
        return df.sort_values(by="Date", ascending=False)
        
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")
        return None

# --- Main App UI ---

st.title("💸 Your Financial Feed")
st.markdown("Load your spending data to generate an interactive dashboard.")

# --- Data Source Selection ---
st.header("1. Choose Your Data Source")

# By default, we select the local file option. The app will try to load it immediately.
source_option = st.radio(
    "Select how to load your spending data:",
    ("Use Default Local File (Android)", "Upload a CSV File"),
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

df = None # Initialize dataframe
if 'filtered_category' not in st.session_state:
    st.session_state.filtered_category = None
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if "messages" not in st.session_state:
    st.session_state.messages = []

if source_option == "Use Default Local File (Android)":
    # This is the filepath for internal storage on Android
    local_filepath = "/storage/emulated/0/Documents/spends.csv"
    df = process_local_spends_file(local_filepath)

elif source_option == "Upload a CSV File":
    with st.expander("ℹ️ Show required CSV format"):
        st.markdown("""
        Your CSV file should have the following columns for best results:
        - `Date` or `DateTime`: The date and time of the transaction.
        - `Amount`: The numerical value of the transaction.
        - `Merchant` or `Recipient (Merchant)`: The name of the merchant.
        - `Type` or `Debit/Credit` (Optional): Should contain 'debit' for spending. If not provided, all transactions are considered debits.
        
        **Example:**
        """)
        example_df = pd.DataFrame([
            {'Date': '2025-07-12 18:30:00', 'Amount': 250.75, 'Merchant': 'Zomato', 'Type': 'debit'},
            {'Date': '2025-07-11 11:00:00', 'Amount': 1200.00, 'Merchant': 'Amazon', 'Type': 'debit'},
        ])
        st.dataframe(example_df, use_container_width=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file)


# --- Dashboard Display (only if data is loaded) ---
if df is not None and not df.empty:
    st.success(f"Successfully loaded {len(df)} transactions. Your dashboard is ready!")
    st.header("2. Your Analytical Dashboard")
    st.markdown("---")
    
    main_content = st.container()
    chat_placeholder = st.empty()

    # --- The Analytical Dashboard: Financial Feed ---
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
            else:
                st.write("No spending recorded yet today.")
        st.markdown("---")

        if st.session_state.filtered_category:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Showing transactions for: **#{st.session_state.filtered_category}**")
            with col2:
                if st.button("Clear Filter", use_container_width=True):
                    st.session_state.filtered_category = None
                    st.rerun()
            display_df = df[df['Category'] == st.session_state.filtered_category]
        else:
            display_df = df

        st.markdown("### Recent Activity")
        last_date = None
        for index, row in display_df.head(50).iterrows():
            current_date = row['Date'].date()
            
            if last_date and last_date.isocalendar()[1] != current_date.isocalendar()[1] and current_date.weekday() == 6:
                with st.container(border=True):
                    st.subheader("🗓️ Weekly Roll-up")
                    week_number = last_date.isocalendar()[1]
                    week_df = df[df['Date'].dt.isocalendar().week == week_number]
                    prev_week_df = df[df['Date'].dt.isocalendar().week == week_number - 1]
                    
                    total_spent_week = week_df['Amount'].sum()
                    total_spent_prev_week = prev_week_df['Amount'].sum()
                    delta = total_spent_week - total_spent_prev_week
                    st.metric(f"Week {week_number} Spending", f"₹{total_spent_week:,.2f}", f"₹{delta:,.2f} vs last week")
                    
                    if not week_df.empty:
                        top_cat = week_df.groupby('Category')['Amount'].sum().nlargest(1).index[0]
                        st.write(f"Your top spending category was **{top_cat}**.")
            
            if current_date != last_date:
                st.markdown(f"**{current_date.strftime('%A, %B %d, %Y')}**")
                last_date = current_date

            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{row['Merchant']}**")
                    if st.button(row['Category'], key=f"cat_{index}", type="secondary"):
                        st.session_state.filtered_category = row['Category']
                        st.rerun()
                with col2:
                    st.markdown(f"<div style='text-align: right; color: #D32F2F; font-weight: bold;'>- ₹{row['Amount']:.2f}</div>", unsafe_allow_html=True)
    
    # --- The LLM-Powered Chat (Simulated) ---
    if st.button("🤖 Chat with Your Data"):
        st.session_state.show_chat = not st.session_state.show_chat

    if st.session_state.show_chat:
        with chat_placeholder.container(border=True):
            st.markdown("### 💬 Your Personal Finance Guru")
            st.caption("Ask me about your spending. I'm powered by a simulated local LLM for privacy.")

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            def get_simulated_llm_response(query):
                query = query.lower()
                
                if "how much" in query and "spend on" in query and "last month" in query:
                    merchant = query.split("spend on ")[1].split(" last month")[0].strip().title()
                    last_month_start = (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1)
                    last_month_end = datetime.now().replace(day=1) - timedelta(days=1)
                    last_month_df = df[(df['Date'] >= last_month_start) & (df['Date'] <= last_month_end)]
                    merchant_df = last_month_df[last_month_df['Merchant'].str.contains(merchant, case=False)]
                    if not merchant_df.empty:
                        return f"You spent **₹{merchant_df['Amount'].sum():,.2f}** on **{merchant}** last month across **{len(merchant_df)}** transactions."
                    return f"I couldn't find any spending records for **{merchant}** last month."
                
                elif "expensive purchase" in query and "last week" in query:
                    last_week_df = df[df['Date'] >= (datetime.now() - timedelta(days=7))]
                    if not last_week_df.empty:
                        most_expensive = last_week_df.loc[last_week_df['Amount'].idxmax()]
                        return f"Your most expensive purchase last week was **₹{most_expensive['Amount']:,.2f}** at **{most_expensive['Merchant']}** on {most_expensive['Date'].strftime('%B %d')}."
                    return "I couldn't find any transactions from last week."
                
                elif "where" in query and "spend the most" in query:
                    top_category = df.groupby('Category')['Amount'].sum().idxmax()
                    top_merchant = df.groupby('Merchant')['Amount'].sum().idxmax()
                    return f"Your highest spending category is **{top_category}**. The merchant you've spent the most with is **{top_merchant}**."
                
                return "I can answer questions like 'How much did I spend on Amazon last month?' or 'What was my most expensive purchase last week?'. Try one of those!"

            if prompt := st.chat_input("Ask about your finances..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    response = get_simulated_llm_response(prompt)
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
            
            if len(df_ts) > 30: # Need sufficient data for a meaningful forecast
                model = LinearRegression().fit(df_ts[['time']], df_ts['Amount'])
                future_days = pd.DataFrame({'time': range(df_ts['time'].max() + 1, df_ts['time'].max() + 31)})
                future_days['Date'] = df_ts['Date'].min() + pd.to_timedelta(future_days['time'], unit='d')
                future_days['forecast'] = model.predict(future_days[['time']])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Amount'], mode='lines', name='Actual'))
                fig.add_trace(go.Scatter(x=future_days['Date'], y=future_days['forecast'], mode='lines', name='Forecast', line=dict(dash='dash')))
                fig.update_layout(title="Projected Spending (Next 30 Days)", xaxis_title=None, yaxis_title="Amount (₹)", margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Not enough data for a forecast. At least 30 days of transactions are recommended.")

    with col2:
        with st.container(border=True):
            st.markdown("#### Smart Alerts")
            # Subscription Slayer
            recurring_payments = df[df['Category'] == 'Bills']['Merchant'].value_counts()
            subscriptions = recurring_payments[recurring_payments >= 2] # 2 or more payments
            if not subscriptions.empty:
                st.subheader("Subscription Slayer")
                st.info(f"We've noticed recurring payments to **{subscriptions.index[0]}**. Are you still using this service?")
            else:
                st.subheader("Subscription Slayer")
                st.info("No potential subscriptions found in your 'Bills' category.")
            
            # Budget Shield
            monthly_budget = 50000
            current_month_spending = df[df['Date'].dt.month == datetime.now().month]['Amount'].sum()
            if current_month_spending > monthly_budget:
                st.subheader("Budget Shield")
                st.error(f"You are ₹{current_month_spending - monthly_budget:,.2f} over your monthly budget of ₹{monthly_budget:,.2f}!")
            elif current_month_spending > monthly_budget * 0.8:
                st.subheader("Budget Shield")
                st.warning(f"Heads up! You've spent ₹{current_month_spending:,.2f} this month, over 80% of your ₹{monthly_budget:,.2f} budget.")
            else:
                st.subheader("Budget Shield")
                st.success("On Track! Your spending is within budget for this month.")


else:
    st.info("Awaiting data to build your financial dashboard...")
                    
