import subprocess
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# Check and install dependencies
def install_dependency(package):
    print(f"Installing dependency '{package}'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["streamlit", "pandas", "plotly", "numpy", "openpyxl"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install_dependency(package)

def load_data(file):
    """Load data from an Excel file."""
    xls = pd.ExcelFile(file)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

def load_all_files():
    """Load all Excel files in the current directory."""
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
    """Extract parameters for a microrobot from the dataframe."""
    def get_value(column):
        return df[column].dropna().iloc[0] if column in df and not df[column].dropna().empty else "N/A"
    
    parameters = {
        "Line": get_value('Line'),
        "Magnet Part": {
            "Radius Extern (mm)": get_value('Magnet Radius extern'),
            "Radius Intern (mm)": get_value('Magnet Radius extern.1'),
            "Height (mm)": get_value('Magnet Hight'),
        },
        "Cup Part": {
            "Radius Extern (mm)": get_value('Cup Radius extern'),
            "Radius Intern (mm)": get_value('Cup Radius extern.1'),
            "Height (mm)": get_value('Cup Hight'),
        }
    }
    return parameters

def calculate_sensitivity(df):
    """Calculate sensitivity percentages for various parameters."""
    factors = ['Flow Profile', 'Actuation Angle', 'Magnetic Distance [cm]', 
               'Actuation Mode', "Gravity Force", "Embedding length"]
    
    # Calculate mean deflections for each factor
    mean_deflections = {factor: df.groupby(factor)['Head Deflection Angle [°]'].mean() 
                        for factor in factors if factor in df}
    
    # Calculate peak-to-peak differences
    deflection_differences = {k: np.ptp(v.values) for k, v in mean_deflections.items()}
    total_difference = sum(deflection_differences.values())
    
    # Calculate sensitivity percentages
    return {k: (v / total_difference) * 100 for k, v in deflection_differences.items()}

def extract_detailed_sensitivity(df):
    """Extract detailed sensitivity metrics for a microrobot."""
    sensitivity_factors = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force',
                          'Actuation Mode', 'Flow Profile', 'Embedding length']

    # Calculate mean deflections for each factor
    mean_deflections = {factor: df.groupby(factor)['Head Deflection Angle [°]'].mean()
                        for factor in sensitivity_factors if factor in df}

    # Calculate peak-to-peak differences
    deflection_differences = {k: np.ptp(v.values) for k, v in mean_deflections.items()}
    total_difference = sum(deflection_differences.values())
    
    # Calculate sensitivity percentages
    sensitivity_percentage = {factor: (deflection_differences.get(factor, 0) / total_difference) * 100
                              for factor in sensitivity_factors}

    # Get max and mean deflection
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

def get_parameter_values(all_data):
    """Get unique parameter values from the dataset."""
    if all_data is None:
        return {}
        
    param_columns = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force', 
                     'Actuation Mode', 'Flow Profile', 'Embedding length']
    
    param_values = {}
    for param in param_columns:
        if param in all_data.columns:
            unique_values = sorted([str(x) for x in all_data[param].unique() if pd.notna(x)])
            param_values[param] = unique_values
    
    return param_values

def filter_data_by_params(df, selected_params):
    """Filter data based on selected parameter values."""
    if df is None or not selected_params:
        return df
    
    filtered_df = df.copy()
    
    for param, values in selected_params.items():
        if param in df.columns and values:
            filtered_df = filtered_df[filtered_df[param].astype(str).isin(values)]
    
    return filtered_df

def perform_special_analysis(all_data, param_values, current_microrobot_data=None):
    """Perform special analysis on parameter variation."""
    # Determine which data to use
    if current_microrobot_data is not None and not current_microrobot_data.empty:
        analysis_data = current_microrobot_data.copy()
        microrobot_param_values = get_parameter_values(current_microrobot_data)
        
        if microrobot_param_values:
            param_values = microrobot_param_values
            microrobot_name = current_microrobot_data['Microrobot'].iloc[0]
            data_source = f"current microrobot ({microrobot_name})"
        else:
            data_source = "all microrobots (fallback)"
    else:
        analysis_data = all_data.copy() if all_data is not None else None
        data_source = "all microrobots"
    
    if analysis_data is None or param_values is None or len(param_values) == 0:
        st.warning("No data available for special analysis. Please ensure data is loaded.")
        return
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("🔍 Special Parameter Analysis")
    st.markdown("Analyze how microrobot deflection varies with one parameter while keeping other parameters fixed.")
    
    st.info(f"Using parameter values from {data_source}.")
    
    # Select parameter for x-axis
    available_params = list(param_values.keys())
    x_param = st.selectbox("Select parameter for X-axis:", available_params, key="special_x_param")
    
    # Create filters for fixed parameters
    st.subheader("Fixed Parameter Values")
    st.markdown("Select one value for each parameter (except the X-axis parameter):")
    
    fixed_params = {}
    col1, col2 = st.columns(2)
    
    left_params = [p for p in available_params if p != x_param][:len(available_params)//2]
    right_params = [p for p in available_params if p != x_param][len(available_params)//2:]
    
    with col1:
        for param in left_params:
            if param in param_values and param != x_param:
                values = param_values[param]
                if values:
                    selected = st.selectbox(f"{param}:", values, key=f"special_fixed_{param}")
                    fixed_params[param] = [selected]
    
    with col2:
        for param in right_params:
            if param in param_values and param != x_param:
                values = param_values[param]
                if values:
                    selected = st.selectbox(f"{param}:", values, key=f"special_fixed_{param}")
                    fixed_params[param] = [selected]
    
    # Analyze button
    if st.button("Analyze Parameter Effect", key="special_analyze_btn"):
        # Filter data based on fixed parameters
        filtered_data = analysis_data.copy()
        
        for param, values in fixed_params.items():
            if param in filtered_data.columns and values:
                filtered_data = filtered_data[filtered_data[param].astype(str).isin([str(v) for v in values])]
        
        # Check for sufficient data
        if filtered_data.empty:
            st.error("No data matches the selected parameter values. Please adjust your selections.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        
        # Display analysis if there's data for the selected parameter
        if x_param in filtered_data.columns:
            # Analysis data source
            special_filter_info = """
            ### Analysis Data Source 
            """
            if current_microrobot_data is not None and not current_microrobot_data.empty:
                microrobot_name = current_microrobot_data['Microrobot'].iloc[0]
                special_filter_info += f"This analysis is specifically for **{microrobot_name}** data only."
            else:
                special_filter_info += "This analysis is using data from **all available microrobots**."
                
            st.markdown(special_filter_info)
            
            # Display fixed parameters
            st.subheader("Analysis Configuration")
            fixed_params_df = pd.DataFrame({
                "Parameter": list(fixed_params.keys()),
                "Fixed Value": [v[0] for v in fixed_params.values()]
            })
            st.dataframe(fixed_params_df, use_container_width=True)
            
            # Calculate statistics
            grouped = filtered_data.groupby(x_param)['Head Deflection Angle [°]']
            stats_df = pd.DataFrame({
                'Mean Deflection': grouped.mean(),
                'Max Deflection': grouped.max(),
                'Min Deflection': grouped.min(),
            }).reset_index()
            
            # Create visualization
            st.subheader(f"Mean Deflection by {x_param}")
            
            # Add microrobot name to title if applicable
            title_suffix = ""
            if current_microrobot_data is not None and not current_microrobot_data.empty:
                microrobot_name = current_microrobot_data['Microrobot'].iloc[0]
                title_suffix = f" for {microrobot_name}"
                
            fig = px.bar(
                stats_df, 
                x=x_param, 
                y='Mean Deflection',
                title=f"Variation of Deflection Angle with {x_param}{title_suffix}",
                labels={'Mean Deflection': 'Mean Deflection Angle [°]'},
                color_discrete_sequence=["royalblue"]
            )
            
            # Add min and max markers
            fig.add_scatter(
                x=stats_df[x_param],
                y=stats_df['Max Deflection'],
                mode='markers',
                name='Maximum',
                marker=dict(color='red', size=10)
            )
            
            # fig.add_scatter(
            #     x=stats_df[x_param],
            #     y=stats_df['Min Deflection'],
            #     mode='markers',
            #     name='Minimum',
            #     marker=dict(color='green', size=10)
            # )
            
            # Layout improvements
            fig.update_layout(
                xaxis_title=x_param,
                yaxis_title='Deflection Angle [°]',
                legend_title="Statistics",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Parameter {x_param} not found in the data.")
    
    st.markdown("</div>", unsafe_allow_html=True)

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
    
    # Section toggles in sidebar
    st.sidebar.header("Dashboard Sections")
    if 'show_microrobot_analysis' not in st.session_state:
        st.session_state.show_microrobot_analysis = True
    if 'show_global_analysis' not in st.session_state:
        st.session_state.show_global_analysis = True
    if 'show_special_analysis' not in st.session_state:
        st.session_state.show_special_analysis = True

    st.session_state.show_microrobot_analysis = st.sidebar.toggle(
        "Show Individual Microrobot Analysis", 
        value=st.session_state.show_microrobot_analysis
    )
    st.session_state.show_global_analysis = st.sidebar.toggle(
        "Show Global Analysis", 
        value=st.session_state.show_global_analysis
    )
    st.session_state.show_special_analysis = st.sidebar.toggle(
        "Show Special Parameter Analysis", 
        value=st.session_state.show_special_analysis
    )
    
    # Load all data initially
    all_data, microrobots = load_all_files()
    param_values = get_parameter_values(all_data)
    
    # File uploader
    with st.container():
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 Upload an Excel file for analysis", type=["xlsx"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Parameter filters section
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("🔄 Unified Parameter Values")
    st.markdown("Select parameter values to include in your analysis:")
    
    # Initialize selected parameters
    if 'selected_params' not in st.session_state:
        st.session_state.selected_params = {}
    
    # Parameter selection UI
    selected_params = {}
    if param_values:
        col1, col2 = st.columns(2)
        
        params_left = list(param_values.keys())[:3]
        params_right = list(param_values.keys())[3:]
        
        with col1:
            for param in params_left:
                if param in param_values:
                    st.subheader(f"{param}")
                    all_values = param_values[param]
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
    
        # Apply filter button
        if st.button("Apply Parameter Filters"):
            st.session_state.selected_params = selected_params
            st.success("Filters applied! Analysis will now use only the selected parameter values.")
    else:
        st.info("No data loaded yet. Please upload a file or ensure Excel files are in the current directory.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Store selections in session state
    if 'selected_params' not in st.session_state:
        st.session_state.selected_params = selected_params
    
    # Individual microrobot analysis
    if uploaded_file is not None and st.session_state.show_microrobot_analysis:
        df = load_data(uploaded_file)
        
        # Apply parameter filters
        if st.session_state.selected_params:
            filtered_df = filter_data_by_params(df, st.session_state.selected_params)
            
            if filtered_df.empty:
                st.warning("No data left after applying filters. Please adjust your parameter selections.")
                filtered_df = df
        else:
            filtered_df = df
        
        microrobot_name = filtered_df['Microrobot'].iloc[0]
        parameters = extract_microrobot_parameters(filtered_df)
        
        # Microrobot specifications
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.header("📌 Microrobot Specifications")
        st.text(f"Line Type: {parameters['Line']}")
        
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
        
        # Microrobot analysis
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.header(f"🔬 Microrobot Analysis: {microrobot_name}")
        
        # Key metrics
        max_deflection = filtered_df['Head Deflection Angle [°]'].max()
        min_deflection = filtered_df['Head Deflection Angle [°]'].min()
        mean_deflection = filtered_df['Head Deflection Angle [°]'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Max Deflection", f"{max_deflection:.2f}°")
        col2.metric("📉 Min Deflection", f"{min_deflection:.2f}°")
        col3.metric("📊 Average Deflection", f"{mean_deflection:.2f}°")
        
        # Parameter values used
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
        
        # Visualizations
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
                             title="Deflection vs Embedding Length", color_discrete_sequence=["darkorange"])
                st.plotly_chart(fig6, use_container_width=True)
        
        # Sensitivity pie chart
        sensitivity_data = calculate_sensitivity(filtered_df)
        sensitivity_df = pd.DataFrame(list(sensitivity_data.items()), columns=['Parameter', 'Sensitivity (%)'])
        fig7 = px.pie(sensitivity_df, values='Sensitivity (%)', names='Parameter', 
              title="Microrobot Deflection Sensitivity to Parameters", 
              color_discrete_sequence=["cyan", "magenta", "yellow", "orange", "gray"])
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Global analysis
    if st.session_state.show_global_analysis:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.header("🌍 Global Analysis of Microrobots")
        
        if all_data is not None and len(microrobots) > 1:
            # Apply parameter filters to global data
            if st.session_state.selected_params:
                filtered_all_data = filter_data_by_params(all_data, st.session_state.selected_params)
                
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
                    rectified_df['Rank'] = rectified_df['Max Deflection (°)'].rank(ascending=False).astype(int)
                    cols = ['Microrobot', 'Rank', 'Max Deflection (°)', 'Mean Deflection (°)',
                           'Most Sensitive Parameter', 'Actuation Angle Sensitivity (%)',
                           'Distance Sensitivity (%)', 'Gravity Sensitivity (%)',
                           'Actuation Mode Sensitivity (%)', 'Flow Sensitivity (%)',
                           'Embedding Length Sensitivity (%)']
                    rectified_df = rectified_df[cols].sort_values(by='Max Deflection (°)', ascending=False)
                    
                    st.markdown("<div class='section'>", unsafe_allow_html=True)
                    st.header("📑 Comparative Table of Microrobots")
                    rectified_df.set_index('Microrobot', inplace=True)
                    st.dataframe(rectified_df, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("Not enough data for comparative analysis after filtering.")
            else:
                st.info("After filtering, there are not enough different microrobots for comparison.")
        else:
            st.info("Not enough microrobot data files found for global analysis.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Special parameter analysis section
    if st.session_state.show_special_analysis:
        # Pass current microrobot data if available
        current_microrobot_data = None
        if uploaded_file is not None:
            current_microrobot_data = filtered_df if 'filtered_df' in locals() else df
        
        perform_special_analysis(all_data, param_values, current_microrobot_data)
    
if __name__ == "__main__":
    main()
