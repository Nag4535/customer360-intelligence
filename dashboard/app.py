import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import joblib
import os

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Customer 360 Intelligence Platform",
    page_icon="🛒",
    layout="wide"
)

# ── DATABASE CONNECTION ───────────────────────────────────
@st.cache_data
def load_data():
    engine = create_engine('sqlite:///database/instacart.db')
    rfm       = pd.read_sql('SELECT * FROM rfm_churn', engine)
    orders    = pd.read_sql('SELECT * FROM orders', engine)
    products  = pd.read_sql('SELECT * FROM products', engine)
    user_feat = pd.read_sql('SELECT * FROM user_features', engine)
    return rfm, orders, products, user_feat

rfm, orders, products, user_feat = load_data()

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
st.sidebar.title("Customer 360")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "RFM Segments", "Churn Analysis", 
     "Product Intelligence", "Customer Lookup"]
)

segment_filter = st.sidebar.multiselect(
    "Filter by Segment",
    options=rfm['Segment'].unique(),
    default=rfm['Segment'].unique()
)

risk_filter = st.sidebar.multiselect(
    "Filter by Churn Risk",
    options=['Low Risk', 'Medium Risk', 'High Risk'],
    default=['Low Risk', 'Medium Risk', 'High Risk']
)

# Apply filters
rfm_filtered = rfm[
    (rfm['Segment'].isin(segment_filter)) &
    (rfm['Churn_Risk'].isin(risk_filter))
]

# ── PAGE 1: OVERVIEW ──────────────────────────────────────
if page == "Overview":
    st.title("Customer 360 Intelligence Platform")
    st.markdown("### Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers",
                  f"{rfm['user_id'].nunique():,}")
    with col2:
        st.metric("Total Orders",
                  f"{orders['order_id'].nunique():,}")
    with col3:
        st.metric("Churn Rate",
                  f"{rfm['Churn'].mean()*100:.1f}%")
    with col4:
        st.metric("Avg Orders/User",
                  f"{rfm['total_orders'].mean():.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        seg_counts = rfm['Segment'].value_counts()
        fig = px.pie(
            values=seg_counts.values,
            names=seg_counts.index,
            title='Customer Segments Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_counts = rfm['Churn_Risk'].value_counts()
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title='Churn Risk Distribution',
            color=risk_counts.index,
            color_discrete_map={
                'Low Risk'   : '#C8FF00',
                'Medium Risk': '#FFA502',
                'High Risk'  : '#FF4757'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Order Patterns")
    col1, col2 = st.columns(2)

    with col1:
        dow_counts = orders['order_dow'].value_counts().sort_index()
        dow_map    = {0:'Sat', 1:'Sun', 2:'Mon', 3:'Tue',
                      4:'Wed', 5:'Thu', 6:'Fri'}
        dow_counts.index = dow_counts.index.map(dow_map)
        fig = px.bar(x=dow_counts.index, y=dow_counts.values,
                     title='Orders by Day of Week',
                     color_discrete_sequence=['#C8FF00'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hour_counts = orders['order_hour_of_day'].value_counts().sort_index()
        fig = px.line(x=hour_counts.index, y=hour_counts.values,
                      title='Orders by Hour of Day',
                      color_discrete_sequence=['#FF6B35'])
        st.plotly_chart(fig, use_container_width=True)

# ── PAGE 2: RFM SEGMENTS ──────────────────────────────────
elif page == "RFM Segments":
    st.title("RFM Customer Segmentation")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Segments", "7")
    with col2:
        st.metric("Champions",
                  f"{(rfm['Segment']=='Champion').sum():,}")
    with col3:
        st.metric("At Risk",
                  f"{(rfm['Segment']=='At Risk').sum():,}")

    st.markdown("---")

    # Segment details
    seg_stats = rfm_filtered.groupby('Segment').agg(
        Customers    = ('user_id', 'count'),
        Avg_Orders   = ('total_orders', 'mean'),
        Avg_Reorder  = ('reorder_rate', 'mean'),
        Avg_Days     = ('avg_days_between', 'mean')
    ).round(2).reset_index()

    st.markdown("### Segment Statistics")
    st.dataframe(seg_stats, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg_stats, x='Segment', y='Avg_Orders',
                     title='Avg Orders by Segment',
                     color='Segment',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(seg_stats, x='Segment', y='Avg_Reorder',
                     title='Avg Reorder Rate by Segment',
                     color='Segment',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    # RFM scatter plot
    st.markdown("### RFM Score Distribution")
    fig = px.scatter(
        rfm_filtered.sample(min(5000, len(rfm_filtered))),
        x='R_score', y='F_score',
        color='Segment',
        title='R vs F Score by Segment',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 3: CHURN ANALYSIS ────────────────────────────────
elif page == "Churn Analysis":
    st.title("Churn Analysis & Prediction")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Churn Rate",
                  f"{rfm['Churn'].mean()*100:.1f}%")
    with col2:
        st.metric("High Risk Customers",
                  f"{(rfm['Churn_Risk']=='High Risk').sum():,}")
    with col3:
        st.metric("Medium Risk",
                  f"{(rfm['Churn_Risk']=='Medium Risk').sum():,}")
    with col4:
        st.metric("Low Risk",
                  f"{(rfm['Churn_Risk']=='Low Risk').sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        churn_seg = rfm.groupby('Segment')['Churn'].mean() * 100
        fig = px.bar(
            x=churn_seg.index,
            y=churn_seg.values,
            title='Churn Rate by Segment (%)',
            color=churn_seg.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            rfm, x='Churn_Probability',
            nbins=50,
            title='Churn Probability Distribution',
            color_discrete_sequence=['#FF6B35']
        )
        st.plotly_chart(fig, use_container_width=True)

    # High risk customers table
    st.markdown("### Top 20 High Risk Customers")
    high_risk = rfm[rfm['Churn_Risk']=='High Risk']\
        .sort_values('Churn_Probability', ascending=False)\
        .head(20)[['user_id', 'Segment', 'total_orders',
                   'reorder_rate', 'Churn_Probability', 'Churn_Risk']]
    st.dataframe(high_risk, use_container_width=True)

# ── PAGE 4: PRODUCT INTELLIGENCE ─────────────────────────
elif page == "Product Intelligence":
    st.title("Product Intelligence")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Products",
                  f"{products['product_id'].nunique():,}")
    with col2:
        st.metric("Total Departments",
                  f"{products['department'].nunique():,}")

    st.markdown("---")

    # Top departments
    dept_counts = products.groupby('department')\
                  ['product_id'].count().sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=dept_counts.head(10).values,
            y=dept_counts.head(10).index,
            orientation='h',
            title='Top 10 Departments by Products',
            color_discrete_sequence=['#C8FF00']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            values=dept_counts.head(8).values,
            names=dept_counts.head(8).index,
            title='Department Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    # Favorite departments by segment
    st.markdown("### Favorite Department by Segment")
    fav_dept = rfm.groupby('Segment')['favorite_department']\
               .agg(lambda x: x.mode()[0]).reset_index()
    fav_dept.columns = ['Segment', 'Favorite Department']
    st.dataframe(fav_dept, use_container_width=True)

# ── PAGE 5: CUSTOMER LOOKUP ───────────────────────────────
elif page == "Customer Lookup":
    st.title("Customer Lookup")
    st.markdown("Enter a Customer ID to see their full profile")

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=1,
        max_value=int(rfm['user_id'].max()),
        value=1
    )

    if st.button("Look Up Customer"):
        customer = rfm[rfm['user_id'] == customer_id]

        if len(customer) > 0:
            c = customer.iloc[0]
            st.markdown(f"### Customer #{customer_id} Profile")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Segment",        c['Segment'])
            with col2:
                st.metric("Total Orders",   int(c['total_orders']))
            with col3:
                st.metric("Reorder Rate",   f"{c['reorder_rate']*100:.1f}%")
            with col4:
                st.metric("Churn Risk",     c['Churn_Risk'])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Churn Probability",
                          f"{c['Churn_Probability']*100:.1f}%")
            with col2:
                st.metric("Unique Products", int(c['unique_products']))
            with col3:
                st.metric("Avg Cart Size",   f"{c['avg_cart_size']:.1f}")
            with col4:
                st.metric("Fav Department",  c['favorite_department'])

            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = c['Churn_Probability'] * 100,
                title = {'text': "Churn Risk Score"},
                gauge = {
                    'axis'  : {'range': [0, 100]},
                    'bar'   : {'color': "#FF4757"},
                    'steps' : [
                        {'range': [0,  30], 'color': '#C8FF00'},
                        {'range': [30, 60], 'color': '#FFA502'},
                        {'range': [60,100], 'color': '#FF4757'},
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Customer {customer_id} not found!")

# ── FOOTER ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Nagarajulu Reddy Nalla**")
st.sidebar.markdown("Data Scientist | ML Engineer | Power BI")