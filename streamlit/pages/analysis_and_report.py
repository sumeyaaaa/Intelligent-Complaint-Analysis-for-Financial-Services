import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
from fpdf import FPDF
import io
import matplotlib.pyplot as plt

# --- Set a stable project root ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# --- Streamlit Setup ---
st.set_page_config(page_title="Analysis and Reports", layout="wide")
st.title("📈 Complaint Data Analysis")
st.markdown("Generate reports and visualize complaint trends.")

# --- Caching to load data only once ---
@st.cache_data
def load_data():
    try:
        data_path = (r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\clean\complaints_clean.csv")
        df = pd.read_csv(data_path)
        df["Date received"] = pd.to_datetime(df["Date received"])
        return df
    except FileNotFoundError:
        st.error(f"Data file not found. Please ensure 'complaints_clean.csv' is in the 'data/clean' directory.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.header("Filter Reports")

    # --- Interactive Widgets ---
    today = datetime.now().date()
    date_range = st.sidebar.date_input(
        "Select a date range:",
        value=(datetime(2023, 1, 1).date(), today),
        min_value=datetime(2020, 1, 1).date(),
        max_value=today
    )

    all_products = df["Product"].unique().tolist()
    selected_products = st.sidebar.multiselect("Filter by Product:", all_products, default=all_products)
    
    all_companies = df["Company"].unique().tolist()
    selected_companies = st.sidebar.multiselect("Filter by Company:", all_companies, default=all_companies)
    
    if "State" in df.columns:
        all_states = df["State"].unique().tolist()
        selected_states = st.sidebar.multiselect("Filter by State:", all_states, default=all_states)
    else:
        st.sidebar.warning("No 'State' column found in data.")
        selected_states = []

    if len(date_range) == 2:
        start_date = date_range[0]
        end_date = date_range[1]
        filtered_df = df[
            (df['Date received'].dt.date >= start_date) & 
            (df['Date received'].dt.date <= end_date) &
            (df['Product'].isin(selected_products)) &
            (df['Company'].isin(selected_companies))
        ]
        if "State" in df.columns:
            filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
    else:
        filtered_df = df.copy()

    # --- Prepare data for charts ---
    product_counts = filtered_df["Product"].value_counts().reset_index()
    product_counts.columns = ["Product", "Count"]

    company_counts = filtered_df["Company"].value_counts().reset_index().head(10)
    company_counts.columns = ["Company", "Count"]

    if "State" in filtered_df.columns:
        state_counts = filtered_df["State"].value_counts().reset_index().head(10)
        state_counts.columns = ["State", "Count"]
    else:
        state_counts = pd.DataFrame(columns=["State", "Count"])

    # --- PDF Generation Function ---
    def create_pdf_report(dataframe, date_range, insights, product_counts, company_counts, state_counts):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Complaint Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Date Range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}", ln=True)
        pdf.cell(0, 10, f"Total Complaints: {len(dataframe)}", ln=True)
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Complaints by Product", ln=True)
        pdf.set_font("Arial", "", 12)
        for index, row in product_counts.iterrows():
            pdf.cell(0, 7, f"- {row['Product']}: {row['Count']}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Top 10 Complaining Companies", ln=True)
        pdf.set_font("Arial", "", 12)
        for index, row in company_counts.iterrows():
            pdf.cell(0, 7, f"- {row['Company']}: {row['Count']}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        if not state_counts.empty:
            pdf.cell(0, 10, "Complaints by State", ln=True)
            pdf.set_font("Arial", "", 12)
            for index, row in state_counts.iterrows():
                pdf.cell(0, 7, f"- {row['State']}: {row['Count']}", ln=True)

        # User Insights Section
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Key Insights & Notes", ln=True)
        pdf.set_font("Arial", "", 12)
        if insights:
            pdf.multi_cell(0, 7, insights)
        else:
            pdf.multi_cell(0, 7, "No insights were added.")

        return pdf.output(dest='S').encode('latin-1')

    # --- Display Metrics and Visualizations ---
    st.subheader("Total Complaints Over Time")
    complaints_over_time = filtered_df.groupby(pd.Grouper(key='Date received', freq='D')).size().reset_index(name='count')
    complaints_over_time.set_index('Date received', inplace=True)
    st.line_chart(complaints_over_time)

    st.subheader("Summary of Complaints")
    st.info(f"Total complaints in selected range: **{len(filtered_df)}**")

    st.subheader("Complaints by Product")
    st.bar_chart(product_counts.set_index("Product"))

    st.subheader("Complaints by Company")
    st.dataframe(company_counts)

    if "State" in filtered_df.columns:
        st.subheader("Complaints by State")
        st.bar_chart(state_counts.set_index("State"))
        
    # --- Add user insights section ---
    st.markdown("---")
    st.subheader("Add Your Insights")
    insights_text = st.text_area("Write your key insights, notes, or observations here. These will be included in the PDF report.")

    # --- Session state to control report generation ---
    if 'pdf_generated' not in st.session_state:
        st.session_state.pdf_generated = False
    
    if st.button("Generate Report"):
        st.session_state.pdf_generated = True

    # --- Download Button (only appears after 'Generate Report' is clicked) ---
    if st.session_state.pdf_generated:
        if not filtered_df.empty:
            pdf_bytes = create_pdf_report(filtered_df, date_range, insights_text, product_counts, company_counts, state_counts)
            st.download_button(
                label="Download Report as PDF",
                data=pdf_bytes,
                file_name="complaint_report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Cannot generate report: No data found in the selected range.")