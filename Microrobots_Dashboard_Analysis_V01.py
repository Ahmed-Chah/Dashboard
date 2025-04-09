import subprocess
import sys

def install_package(package):
    print(f"⏳ Installation de la dépendance '{package}' en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Liste des dépendances nécessaires
dependencies = ["streamlit", "pandas", "plotly", "numpy", "openpyxl"]

# Vérification et installation dynamique des dépendances
for dependency in dependencies:
    try:
        __import__(dependency)
    except ImportError:
        install_package(dependency)

# Tes imports originaux (ils fonctionneront forcément après l'installation dynamique)
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

def load_data(file):
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    return df

def load_all_files():
    all_data = []
    microrobots = []
    folder_path = os.getcwd()
    
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            df = load_data(os.path.join(folder_path, file))
            microrobot_name = df['Microrobot'].iloc[0]
            df['Microrobot'] = microrobot_name
            all_data.append(df)
            microrobots.append(microrobot_name)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True), microrobots
    return None, None

def extract_microrobot_parameters(df):
    parameters = {
        "Line": df['Line'].dropna().iloc[0] if 'Line' in df and not df['Line'].dropna().empty else "N/A",
        "Magnet Part": {
            "Radius Extern (mm)": df['Magnet Radius extern'].dropna().iloc[0] if 'Magnet Radius extern' in df and not df['Magnet Radius extern'].dropna().empty else "N/A",
            "Radius Intern (mm)": df['Magnet Radius extern.1'].dropna().iloc[0] if 'Magnet Radius extern.1' in df and not df['Magnet Radius extern.1'].dropna().empty else "N/A",
            "Height (mm)": df['Magnet Hight'].dropna().iloc[0] if 'Magnet Hight' in df and not df['Magnet Hight'].dropna().empty else "N/A",
        },
        "Cup Part": {
            "Radius Extern (mm)": df['Cup Radius extern'].dropna().iloc[0] if 'Cup Radius extern' in df and not df['Cup Radius extern'].dropna().empty else "N/A",
            "Radius Intern (mm)": df['Cup Radius extern.1'].dropna().iloc[0] if 'Cup Radius extern.1' in df and not df['Cup Radius extern.1'].dropna().empty else "N/A",
            "Height (mm)": df['Cup Hight'].dropna().iloc[0] if 'Cup Hight' in df and not df['Cup Hight'].dropna().empty else "N/A",
        }
    }
    return parameters

# Fonction pour extraire dynamiquement les sensibilités détaillées
def extract_detailed_sensitivity(df):
    sensitivity_factors = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force',
                           'Actuation Mode', 'Flow Profile', 'Embedding length']

    mean_deflections = {factor: df.groupby(factor)['Head Deflection Angle [°]'].mean()
                        for factor in sensitivity_factors if factor in df}

    deflection_differences = {k: np.ptp(v.values) for k, v in mean_deflections.items()}
    total_difference = sum(deflection_differences.values())
    sensitivity_percentage = {factor: (deflection_differences.get(factor, 0) / total_difference) * 100
                              for factor in sensitivity_factors}

    max_deflection = df['Head Deflection Angle [°]'].max()
    mean_deflection = df['Head Deflection Angle [°]'].mean()

    return {
        "Microrobot": df['Microrobot'].iloc[0],
        "Max Deflection (°)": round(max_deflection, 2),
        "Mean Deflection (°)": round(mean_deflection, 2),
        "Most Sensitive Parameter": max(sensitivity_percentage, key=sensitivity_percentage.get),
        "Actuation Angle Sensitivity (%)": round(sensitivity_percentage['Actuation Angle'], 2),
        "Distance Sensitivity (%)": round(sensitivity_percentage['Magnetic Distance [cm]'], 2),
        "Gravity Sensitivity (%)": round(sensitivity_percentage['Gravity Force'], 2),
        "Actuation Mode Sensitivity (%)": round(sensitivity_percentage['Actuation Mode'], 2),
        "Flow Sensitivity (%)": round(sensitivity_percentage['Flow Profile'], 2),
        "Embedding Length Sensitivity (%)": round(sensitivity_percentage['Embedding length'], 2)
    }

def calculate_sensitivity(df):
    factors = ['Flow Profile', 'Actuation Angle', 'Magnetic Distance [cm]', 'Actuation Mode', "Gravity Force", "Embedding length"]
    mean_deflections = {factor: df.groupby(factor)['Head Deflection Angle [°]'].mean() for factor in factors}
    deflection_differences = {k: np.ptp(v.values) for k, v in mean_deflections.items()}
    total_difference = sum(deflection_differences.values())
    sensitivity_percentage = {k: (v / total_difference) * 100 for k, v in deflection_differences.items()}
    return sensitivity_percentage

# New function to get all unique parameter values from all data
def get_parameter_values(all_data):
    if all_data is None:
        return {}
        
    # Define parameter columns we're interested in
    param_columns = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force', 
                     'Actuation Mode', 'Flow Profile', 'Embedding length']
    
    # Get unique values for each parameter
    param_values = {}
    for param in param_columns:
        if param in all_data.columns:
            # Convert all values to strings before sorting to avoid type comparison issues
            unique_values = sorted([str(x) for x in all_data[param].unique() if pd.notna(x)])
            param_values[param] = unique_values
    
    return param_values

# Function to filter data based on selected parameter values
def filter_data_by_params(df, selected_params):
    if df is None or not selected_params:
        return df
    
    # Start with the original dataframe
    filtered_df = df.copy()
    
    # Apply filters for each parameter
    for param, values in selected_params.items():
        if param in df.columns and values:
            # Convert values in the dataframe to strings for comparison
            filtered_df = filtered_df[filtered_df[param].astype(str).isin(values)]
    
    return filtered_df

def main():
    st.set_page_config(page_title="Microrobot Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .main {
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stMetric {
            font-size: 18px;
            font-weight: bold;
        }
        .section {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Microrobot Analysis Dashboard")
    
    # Load all data at the beginning to get parameter values
    all_data, microrobots = load_all_files()
    param_values = get_parameter_values(all_data)
    
    with st.container():
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 Upload an Excel file for analysis", type=["xlsx"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    # NEW SECTION: Parameter value filters
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("🔄 Unified Parameter Values")
    st.markdown("Select which parameter values to include in your analysis for consistent comparison across microrobots:")
    
    # Initialize selected parameters dictionary
    if 'selected_params' not in st.session_state:
        st.session_state.selected_params = {}
    
    # Create UI for parameter selection
    selected_params = {}
    if param_values:
        col1, col2 = st.columns(2)
        
        params_left = list(param_values.keys())[:3]  # First half of parameters
        params_right = list(param_values.keys())[3:]  # Second half of parameters
        
        with col1:
            for param in params_left:
                if param in param_values:
                    st.subheader(f"{param}")
                    # Get all unique values for this parameter
                    all_values = param_values[param]
                    # Create a multiselect for each parameter
                    selected_values = st.multiselect(f"Select values for {param}", 
                                                    options=all_values,
                                                    default=all_values,
                                                    key=f"select_{param}")
                    selected_params[param] = selected_values
        
        with col2:
            for param in params_right:
                if param in param_values:
                    st.subheader(f"{param}")
                    all_values = param_values[param]
                    selected_values = st.multiselect(f"Select values for {param}", 
                                                    options=all_values,
                                                    default=all_values,
                                                    key=f"select_{param}")
                    selected_params[param] = selected_values
    
        # Button to apply filters
        if st.button("Apply Parameter Filters"):
            st.session_state.selected_params = selected_params
            st.success("Filters applied! Analysis will now use only the selected parameter values.")
    else:
        st.info("No data loaded yet. Please upload a file or ensure Excel files are in the current directory.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Store the current selection in session state
    if 'selected_params' not in st.session_state:
        st.session_state.selected_params = selected_params
    
    # Individual file analysis
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Apply parameter filters
        if st.session_state.selected_params:
            filtered_df = filter_data_by_params(df, st.session_state.selected_params)
            
            # Check if we still have data after filtering
            if filtered_df.empty:
                st.warning("No data left after applying filters. Please adjust your parameter selections.")
                # Use original data in this case
                filtered_df = df
        else:
            filtered_df = df
        
        microrobot_name = filtered_df['Microrobot'].iloc[0]
        parameters = extract_microrobot_parameters(filtered_df)
        
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.header("📌 Microrobot Specifications")
        st.text(f"**Line Type : {parameters['Line']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧲 Magnet Part")
            for key, value in parameters['Magnet Part'].items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.subheader("🔵 Cup Part")
            for key, value in parameters['Cup Part'].items():
                st.text(f"{key}: {value}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.header(f"🔬 Microrobot Analysis: {microrobot_name}")
        
        max_deflection = filtered_df['Head Deflection Angle [°]'].max()
        min_deflection = filtered_df['Head Deflection Angle [°]'].min()
        mean_deflection = filtered_df['Head Deflection Angle [°]'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Max Deflection", f"{max_deflection:.2f}°")
        col2.metric("📉 Min Deflection", f"{min_deflection:.2f}°")
        col3.metric("📊 Average Deflection", f"{mean_deflection:.2f}°")
        
        # Show the parameter values being used
        st.markdown("### 🔍 Parameter Values Used in Current Analysis")
        param_columns = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force', 
                         'Actuation Mode', 'Flow Profile', 'Embedding length']
        
        param_usage = {}
        for param in param_columns:
            if param in filtered_df.columns:
                param_usage[param] = sorted(filtered_df[param].unique())
        
        param_df = pd.DataFrame({"Parameter": param_usage.keys(), 
                                 "Values Used": [str(v) for v in param_usage.values()]})
        st.dataframe(param_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Magnetic Distance [cm]' in filtered_df.columns:
                fig1 = px.box(filtered_df, x='Magnetic Distance [cm]', y='Head Deflection Angle [°]', 
                             title="Deflection vs Distance", color_discrete_sequence=["blue"])
                st.plotly_chart(fig1, use_container_width=True)
        with col2:
            if 'Actuation Angle' in filtered_df.columns:
                fig2 = px.box(filtered_df, x='Actuation Angle', y='Head Deflection Angle [°]', 
                             title="Deflection vs Actuation Angle", color_discrete_sequence=["red"])
                st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            if 'Flow Profile' in filtered_df.columns:
                fig3 = px.box(filtered_df, x='Flow Profile', y='Head Deflection Angle [°]', 
                             title="Deflection vs Flow Profile", color_discrete_sequence=["green"])
                st.plotly_chart(fig3, use_container_width=True)
        with col4:
            if 'Actuation Mode' in filtered_df.columns:
                fig4 = px.box(filtered_df, x='Actuation Mode', y='Head Deflection Angle [°]', 
                             title="Deflection vs Actuation Mode", color_discrete_sequence=["purple"])
                st.plotly_chart(fig4, use_container_width=True)
            
        col5, col6 = st.columns(2)
        with col5:
            if 'Gravity Force' in filtered_df.columns:
                fig5 = px.box(filtered_df, x='Gravity Force', y='Head Deflection Angle [°]', 
                             title="Deflection vs Gravity", color_discrete_sequence=["chocolate"])
                st.plotly_chart(fig5, use_container_width=True)
        with col6:
            if 'Embedding length' in filtered_df.columns:
                fig6 = px.box(filtered_df, x='Embedding length', y='Head Deflection Angle [°]', 
                             title="Deflection vs Embedding length", color_discrete_sequence=["darkorange"])
                st.plotly_chart(fig6, use_container_width=True)
        
        sensitivity_data = calculate_sensitivity(filtered_df)
        sensitivity_df = pd.DataFrame(list(sensitivity_data.items()), columns=['Parameter', 'Sensitivity (%)'])
        fig7 = px.pie(sensitivity_df, values='Sensitivity (%)', names='Parameter', 
              title="Microrobot Deflection Sensitivity to Parameters", 
              color_discrete_sequence=["cyan", "magenta", "yellow", "orange", "gray"])
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Global Analysis with filtered data
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("🌍 Global Analysis of Microrobots")
    
    if all_data is not None and len(microrobots) > 1:
        # Apply parameter filters to global data
        if st.session_state.selected_params:
            filtered_all_data = filter_data_by_params(all_data, st.session_state.selected_params)
            
            # Check if we still have data after filtering
            if filtered_all_data.empty:
                st.warning("No global data left after applying filters. Please adjust your parameter selections.")
                filtered_all_data = all_data
        else:
            filtered_all_data = all_data
        
        # Check if we have multiple microrobots after filtering
        if len(filtered_all_data['Microrobot'].unique()) > 1:
            fig8 = px.box(filtered_all_data, x='Microrobot', y='Head Deflection Angle [°]', 
                         title="Deflection by Microrobot", color_discrete_sequence=["brown"])
            st.plotly_chart(fig8, use_container_width=True)
            
            max_deflection_by_micro = filtered_all_data.groupby('Microrobot')['Head Deflection Angle [°]'].max().reset_index()
            max_deflection_by_micro = max_deflection_by_micro.sort_values(by='Head Deflection Angle [°]', ascending=True).head(20)
            fig9 = px.bar(max_deflection_by_micro, x='Microrobot', y='Head Deflection Angle [°]', 
                         title="Top Microrobots by Max Deflection", color_discrete_sequence=["darkblue"])
            st.plotly_chart(fig9, use_container_width=True)
            
            # Comparative table with filtered data
            microrobots_after_filter = filtered_all_data['Microrobot'].unique()
            all_sensitivity_data = []
            
            for robot in microrobots_after_filter:
                robot_data = filtered_all_data[filtered_all_data['Microrobot'] == robot]
                if not robot_data.empty:
                    try:
                        sensitivity = extract_detailed_sensitivity(robot_data)
                        all_sensitivity_data.append(sensitivity)
                    except Exception as e:
                        st.warning(f"Could not extract sensitivity for {robot}: {e}")
            
            if all_sensitivity_data:
                rectified_df = pd.DataFrame(all_sensitivity_data)
                rectified_df['Classement'] = rectified_df['Max Deflection (°)'].rank(ascending=False).astype(int)
                rectified_df = rectified_df[['Microrobot', 'Classement', 'Max Deflection (°)', 'Mean Deflection (°)',
                                             'Most Sensitive Parameter', 'Actuation Angle Sensitivity (%)',
                                             'Distance Sensitivity (%)', 'Gravity Sensitivity (%)',
                                             'Actuation Mode Sensitivity (%)', 'Flow Sensitivity (%)',
                                             'Embedding Length Sensitivity (%)']].sort_values(by='Max Deflection (°)', ascending=False)
                
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.header("📑 Comparative Table of Microrobots vs Params")
                # On définit la colonne "Microrobot" comme index pour rester visible en scrolling horizontal
                rectified_df.set_index('Microrobot', inplace=True)
                st.dataframe(rectified_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Not enough data for comparative analysis after filtering. Try adjusting your parameter selections.")
        else:
            st.info("After filtering, there are not enough different microrobots for comparison. Try adjusting your parameter selections.")
    else:
        st.info("Not enough microrobot data files found for global analysis. Please ensure multiple Excel files are available in the current directory.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()