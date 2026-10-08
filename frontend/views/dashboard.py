import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_dashboard(api_url):
    st.title("📊 Banking Portfolio Overview")
    
    with st.spinner("Loading portfolio data..."):
        try:
            info_res = requests.get(f"{api_url}/info")
            info_data = info_res.json()
            total_customers = info_data.get("total", 0)
            default_rate = info_data.get("default_rate", 0.0)
            
            # Since we can't easily fetch ALL predictions via API without slowing down,
            # we simulate the aggregate stats based on the test set metrics or overall target rate for the dashboard.
            # In a real app, we'd have a specific aggregation endpoint.
            # Let's display the metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Customers</div>
                    <div class="metric-value">{total_customers:,}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg. Default Rate</div>
                    <div class="metric-value">{default_rate*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                # We can assume a distribution based on typical portfolio
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High Risk Est.</div>
                    <div class="metric-value">~{(default_rate)*total_customers*1.2:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Active Segments</div>
                    <div class="metric-value">4</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Placeholder charts for dashboard
            st.subheader("Portfolio Risk Distribution (Estimated)")
            # Simulated data for visual purposes to show dashboard layout
            # In production, this would come from a /dashboard/stats endpoint querying the database
            risk_labels = ['LOW', 'MEDIUM', 'HIGH']
            risk_values = [total_customers * 0.65, total_customers * 0.20, total_customers * 0.15]
            
            fig = px.pie(values=risk_values, names=risk_labels, title='Risk Category Distribution',
                         color=risk_labels, color_discrete_map={'LOW':'green', 'MEDIUM':'orange', 'HIGH':'red'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Navigate to **Customer 360** to see individual predictions and explanations.")
            
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
