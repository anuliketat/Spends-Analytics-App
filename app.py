import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

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
        st.error("The 'style.css' file was not found. Please ensure it's in the same directory as 'app.py'.")

local_css("style.css")

# --- Data Generation (Simulating SMS Data) ---
@st.cache_data
def generate_synthetic_data():
    """Generates a realistic-looking DataFrame of personal finance transactions."""
    # Use the current date to make the data feel relevant
    today = datetime.now()
    start_date = today - timedelta(days=90)
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=today, freq='D'))
    
    data = []
    categories = {
        "Food": ["Zomato", "Swiggy", "Local Restaurant", "Grocery Store"],
        "Transport": ["Uber", "Ola", "Metro Card", "Fuel Station"],
        "Shopping": ["Amazon", "Flipkart", "Myntra", "Zara"],
        "Bills": ["Airtel", "Vodafone", "Electricity Board", "Credit Card"],
        "Entertainment": ["Netflix", "Spotify", "BookMyShow", "PVR Cinemas"],
        "Health": ["Apollo Pharmacy", "Netmeds", "Local Clinic"],
        "Other": ["ATM Withdrawal", "Miscellaneous"]
    }
    
    for date in date_range:
        num_transactions = np.random.randint(2, 6) if date.weekday() >= 5 else np.random.randint(1, 4)
        for _ in range(num_transactions):
            category = np.random.choice(list(categories.keys()), p=[0.3, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05])
            merchant = np.random.choice(categories[category])
            
            amount_ranges = {
                "Bills": (500, 5000), "Health": (500, 5000),
                "Shopping": (200, 8000), "Default": (50, 1000)
            }
            min_val, max_val = amount_ranges.get(category, amount_ranges["Default"])
            amount = np.random.uniform(min_val, max_val)
            
            hour = np.random.randint(8, 23)
            transaction_time = date.replace(hour=hour, minute=np.random.randint(0, 59), second=np.random.randint(0, 59))
            
            data.append({"Date": transaction_time, "Merchant": merchant, "Amount": round(amount, 2), "Category": category})
            
    df = pd.DataFrame(data).sort_values(by="Date", ascending=False)
    return df

df = generate_synthetic_data()

# --- Session State Initialization ---
if 'filtered_category' not in st.session_state:
    st.session_state.filtered_category = None
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
def filter_by_category(category):
    st.session_state.filtered_category = category

def clear_filter():
    st.session_state.filtered_category = None

# --- UI Components ---
st.title("💸 Your Financial Feed")
st.markdown("A dynamic and interactive summary of your spending.")
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
            st.button("Clear Filter", on_click=clear_filter, use_container_width=True)
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
                    st.markdown(f"**Pro-Tip:** Your spending on '{top_cat}' was highest this week. See if you can optimize!")
        
        if current_date != last_date:
            st.markdown(f"**{current_date.strftime('%A, %B %d, %Y')}**")
            last_date = current_date

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{row['Merchant']}**")
                category_class = f"category-{row['Category'].lower()}"
                if st.button(row['Category'], key=f"cat_{index}", type="secondary"):
                    filter_by_category(row['Category'])
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
        
        if len(df_ts) > 1:
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
            st.write("Not enough data for a forecast.")

with col2:
    with st.container(border=True):
        st.markdown("#### Smart Alerts")
        recurring_payments = df[df['Category'] == 'Bills']['Merchant'].value_counts()
        subscriptions = recurring_payments[recurring_payments >= 3]
        if not subscriptions.empty:
            st.subheader("Subscription Slayer")
            st.info(f"We've noticed recurring payments to **{subscriptions.index[0]}**. Are you still using this service?")
        
        monthly_budget = 50000
        current_month_spending = df[df['Date'].dt.month == datetime.now().month]['Amount'].sum()
        if current_month_spending > monthly_budget * 0.8:
            st.subheader("Budget Shield")
            st.warning(f"Heads up! You've spent ₹{current_month_spending:,.2f} this month, over 80% of your ₹{monthly_budget:,.2f} budget.")
          
