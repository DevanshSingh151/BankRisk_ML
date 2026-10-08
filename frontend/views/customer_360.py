import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_customer_360(api_url):
    st.title("🧑‍💼 Customer 360 View")
    
    # Customer Selection
    st.markdown("### Search Customer")
    col_search, _ = st.columns([1, 2])
    with col_search:
        # Default dataset has 30,000 customers. Let's provide a numeric input
        customer_id = st.number_input("Enter Customer ID (1 - 30000):", min_value=1, max_value=30000, value=124)
        search_btn = st.button("Load Profile", type="primary")
        
    if search_btn or 'current_customer' not in st.session_state or st.session_state.current_customer != customer_id:
        st.session_state.current_customer = customer_id
        
        with st.spinner("Fetching comprehensive customer profile..."):
            try:
                # 1. Get raw data
                raw_res = requests.get(f"{api_url}/customer/{customer_id}")
                if raw_res.status_code != 200:
                    st.error("Customer not found or API error.")
                    return
                cust_data = raw_res.json()
                
                # 2. Get Credit Risk
                cr_res = requests.post(f"{api_url}/predict/credit-risk", json={"customer_id": customer_id}).json()
                
                # 3. Get Delinquency Risk
                dw_res = requests.post(f"{api_url}/predict/delinquency", json={"customer_id": customer_id}).json()
                
                # 4. Get Segment
                seg_res = requests.get(f"{api_url}/customer/{customer_id}/segment").json()
                
                # 5. Get Recommendation
                rec_res = requests.get(f"{api_url}/customer/{customer_id}/recommendation").json()
                
                # 6. Get Explainability
                exp_res = requests.get(f"{api_url}/predict/credit-risk/explain/{customer_id}").json()
                
                # Save to session state
                st.session_state.customer_data = {
                    "raw": cust_data,
                    "credit_risk": cr_res,
                    "delinquency": dw_res,
                    "segment": seg_res,
                    "recommendation": rec_res,
                    "explain": exp_res
                }
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return
                
    if 'customer_data' in st.session_state:
        data = st.session_state.customer_data
        
        # Header Metrics
        st.markdown(f"### Profile Overview: Customer #{st.session_state.current_customer}")
        
        cr = data['credit_risk']
        seg = data['segment']
        
        st.markdown("---")
        
        # Tabs for different modules
        tab1, tab2, tab3 = st.tabs(["Credit Risk & Explainability", "Early Warning & Behavior", "Next Best Offer & Action"])
        
        with tab1:
            col_risk, col_explain = st.columns([1, 2])
            
            with col_risk:
                st.subheader("Credit Risk Assessment")
                st.markdown(f"**Default Probability:** {cr['probability']*100:.1f}%")
                st.markdown(f"**BankRisk360 Score:** {cr['risk_score']}/100")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = cr['risk_score'],
                    title = {'text': "Risk Score (Higher is Better)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},
                            {'range': [50, 75], 'color': "moccasin"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            with col_explain:
                st.subheader("Why this prediction? (SHAP)")
                exp = data['explain']
                
                # Create waterfall data
                features = []
                contributions = []
                
                # Base value
                features.append("Base Probability")
                contributions.append(exp['base_value'])
                
                for f in exp['top_risk_drivers']:
                    features.append(f"{f['feature']} ({f['value']:.2f})")
                    contributions.append(f['contribution'])
                    
                for f in exp['top_risk_reducers']:
                    features.append(f"{f['feature']} ({f['value']:.2f})")
                    contributions.append(f['contribution'])
                    
                fig2 = go.Figure(go.Waterfall(
                    name = "SHAP", orientation = "h",
                    measure = ["absolute"] + ["relative"] * (len(features)-1),
                    y = features,
                    x = contributions,
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                ))
                fig2.update_layout(title="Top Factors Influencing Default Risk", height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)
                
        with tab2:
            st.subheader("Early Warning / Delinquency Risk")
            dw = data['delinquency']
            
            dw_color = "red" if dw['risk_category'] == "HIGH" else "orange" if dw['risk_category'] == "MEDIUM" else "green"
            st.markdown(f"**Short-term Delinquency Probability:** {dw['probability']*100:.1f}%")
            st.markdown(f"**Risk Tier:** <span style='color:{dw_color}; font-weight:bold;'>{dw['risk_category']}</span>", unsafe_allow_html=True)
            
            st.markdown("### Behavioral History")
            # Plot payment history
            hist_cols = ['pay_6', 'pay_5', 'pay_4', 'pay_3', 'pay_2', 'pay_0']
            months = ['Month -6', 'Month -5', 'Month -4', 'Month -3', 'Month -2', 'Current']
            delays = [data['raw'].get(c, 0) for c in hist_cols]
            
            fig_hist = px.bar(x=months, y=delays, labels={'x': 'Time', 'y': 'Months Delayed'}, title="Repayment Delay History")
            fig_hist.update_traces(marker_color=['red' if d >= 2 else 'orange' if d == 1 else 'green' for d in delays])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab3:
            st.subheader("Decision Support Engine")
            rec = data['recommendation']
            
            st.markdown("### Next-Best-Offer Recommendation")
            st.info(f"**Recommended Product:** {rec['recommended_product']} (Confidence: {rec['confidence']*100:.0f}%)")
            st.markdown("**Reasons for recommendation:**")
            for r in rec['reasons']:
                st.markdown(f"- {r}")
                
            st.markdown("---")
            st.markdown("### Recommended Bank Actions")
            st.warning(f"**Credit Risk Action:** {cr['action']}")
            if dw['risk_category'] != "LOW":
                st.error(f"**Early Warning Action:** {dw['action']}")
