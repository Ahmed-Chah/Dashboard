import subprocess
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import uuid

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

def show_parameter_variations(df, all_data=None):
    """
    Extract and display the parameter test matrix for each microrobot.
    
    Args:
        df: DataFrame containing data for a single microrobot
        all_data: DataFrame containing data for all microrobots (optional)
    """
    if df is None or df.empty:
        return
    
    # Get microrobot name
    microrobot_name = df['Microrobot'].iloc[0]
    
    # Define parameters to analyze
    params = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force', 
              'Actuation Mode', 'Flow Profile', 'Embedding length']
    
    with st.expander(f"🧪 Test Matrix for {microrobot_name}", expanded=False):
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown(f"### Parameter Variations for {microrobot_name}")
        st.markdown("This table shows what parameter values were tested for this microrobot:")
        
        # Create data for the table
        test_matrix_data = []
        
        for param in params:
            if param in df.columns:
                # Extract unique values for this parameter for this microrobot
                unique_values = sorted([str(val) for val in df[param].unique() if pd.notna(val)])
                
                # Compare with global variations if all_data is provided
                global_unique = []
                if all_data is not None and param in all_data.columns:
                    global_unique = sorted([str(val) for val in all_data[param].unique() if pd.notna(val)])
                
                # Check if this microrobot has custom/unique variations
                has_unique_variations = False
                if global_unique and set(unique_values) != set(global_unique):
                    has_unique_variations = True
                
                # Add row to the table
                test_matrix_data.append({
                    "Parameter": param,
                    "Values Tested": ", ".join(unique_values),
                    "Number of Values": len(unique_values),
                    "Has Unique Variations": "✓" if has_unique_variations else ""
                })
        
        # Create and display the test matrix table
        if test_matrix_data:
            test_matrix_df = pd.DataFrame(test_matrix_data)
            st.dataframe(test_matrix_df, use_container_width=True)
            
            # Add a visualization of parameter variations
            st.markdown("### 📊 Parameter Variations Visualization")
            st.markdown("Select a parameter to see how its values are distributed in the tests:")
            
            # Parameter selection
            selected_param = st.selectbox(
                "Select Parameter:",
                [p for p in params if p in df.columns],
                key="param_var_select"
            )
            
            if selected_param in df.columns:
                # Count occurrences of each value
                value_counts = df[selected_param].value_counts().reset_index()
                value_counts.columns = [selected_param, 'Count']
                
                # Create bar chart
                fig = px.bar(
                    value_counts, 
                    x=selected_param, 
                    y='Count',
                    title=f"Test Count for Each {selected_param} Value",
                    color_discrete_sequence=["teal"]
                )
                st.plotly_chart(fig, use_container_width=True, key="param_var_chart")
        else:
            st.info("No parameter variations found for this microrobot.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_simple_parameter_variations_table(all_data):
    """
    Create a simple table showing parameter variations for all microrobots.
    
    Args:
        all_data: DataFrame containing data for all microrobots
    """
    if all_data is None or all_data.empty:
        st.warning("No microrobot data available for analysis.")
        return
    
    # Get list of all microrobots
    microrobots = sorted(all_data['Microrobot'].unique())
    
    # Define parameters to analyze
    param_columns = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force', 
                     'Actuation Mode', 'Flow Profile', 'Embedding length']
    
    with st.expander("📊 Parameter Variations for All Microrobots", expanded=True):
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("This table shows what parameter values were tested for each microrobot design.")
        
        # Create a table with all parameters directly included
        variation_data = []
        
        for robot in microrobots:
            robot_data = all_data[all_data['Microrobot'] == robot]
            robot_variations = {"Microrobot": robot}
            
            for param in param_columns:
                if param in robot_data.columns:
                    unique_values = sorted([
                        str(val) for val in robot_data[param].unique() 
                        if pd.notna(val)
                    ])
                    
                    robot_variations[param] = ", ".join(unique_values)
                else:
                    robot_variations[param] = "N/A"
            
            variation_data.append(robot_variations)
        
        if variation_data:
            variation_df = pd.DataFrame(variation_data)
            
            # Set Microrobot column as index to fix it when scrolling
            variation_df = variation_df.set_index('Microrobot')
            
            # Display the table with all parameters
            st.dataframe(variation_df, use_container_width=True)
        else:
            st.info("No parameter variation data available.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_all_microrobots_variations(all_data):
    """
    Create a comprehensive table showing parameter variations for all microrobots.
    
    Args:
        all_data: DataFrame containing data for all microrobots
    """
    if all_data is None or all_data.empty:
        st.warning("No microrobot data available for analysis.")
        return
    
    # Get list of all microrobots
    microrobots = sorted(all_data['Microrobot'].unique())
    
    # Define parameters to analyze
    param_columns = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force', 
                     'Actuation Mode', 'Flow Profile', 'Embedding length']
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("📊 Parameter Variations Across All Microrobots")
    st.markdown("This section shows what parameter values were tested for each microrobot design.")
    
    # Create tabs for different views of the data
    tab1, tab2, tab3 = st.tabs(["Parameter Matrix", "Variation Comparison", "Detailed View"])
    
    with tab1:
        # First, create a big matrix showing parameter values for each microrobot
        st.subheader("Parameter Test Matrix")
        
        # Select parameter to analyze
        selected_param = st.selectbox(
            "Select parameter to view across all microrobots:",
            options=param_columns,
            index=0,
            key="matrix_param_select"
        )
        
        if selected_param not in all_data.columns:
            st.warning(f"Parameter {selected_param} not found in the data.")
        else:
            # Create a table structure
            matrix_data = []
            
            for robot in microrobots:
                robot_data = all_data[all_data['Microrobot'] == robot]
                if not robot_data.empty and selected_param in robot_data.columns:
                    values = sorted([
                        str(val) for val in robot_data[selected_param].unique() 
                        if pd.notna(val)
                    ])
                    
                    matrix_data.append({
                        "Microrobot": robot,
                        "Values Tested": ", ".join(values),
                        "Number of Values": len(values)
                    })
            
            if matrix_data:
                matrix_df = pd.DataFrame(matrix_data)
                st.dataframe(matrix_df, use_container_width=True)
                
                # Visualization of test coverage
                fig = px.bar(
                    matrix_df,
                    x="Microrobot",
                    y="Number of Values",
                    title=f"Number of {selected_param} Values Tested by Each Microrobot",
                    color_discrete_sequence=["teal"]
                )
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True, key="matrix_coverage_chart")
            else:
                st.info(f"No data available for parameter {selected_param}.")
    
    with tab2:
        st.subheader("Parameter Variation Comparison")
        st.markdown("Compare how different parameters are varied across microrobots.")
        
        # Create summary of parameter variation coverage
        coverage_data = []
        
        for param in param_columns:
            if param in all_data.columns:
                # Count how many microrobots have each number of variations
                variation_counts = {}
                
                for robot in microrobots:
                    robot_data = all_data[all_data['Microrobot'] == robot]
                    if not robot_data.empty and param in robot_data.columns:
                        unique_values = len([
                            val for val in robot_data[param].unique() 
                            if pd.notna(val)
                        ])
                        
                        if unique_values in variation_counts:
                            variation_counts[unique_values] += 1
                        else:
                            variation_counts[unique_values] = 1
                
                # Calculate statistics
                if variation_counts:
                    max_variations = max(variation_counts.keys())
                    min_variations = min(variation_counts.keys())
                    total_robots = sum(variation_counts.values())
                    avg_variations = sum([k * v for k, v in variation_counts.items()]) / total_robots
                    
                    coverage_data.append({
                        "Parameter": param,
                        "Min Values": min_variations,
                        "Max Values": max_variations,
                        "Avg Values": round(avg_variations, 1),
                        "Most Common": max(variation_counts.items(), key=lambda x: x[1])[0]
                    })
        
        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)
            st.dataframe(coverage_df, use_container_width=True)
            
            # Bar chart comparing parameter variation ranges
            fig = go.Figure()
            
            for param in coverage_df["Parameter"]:
                param_data = coverage_df[coverage_df["Parameter"] == param].iloc[0]
                fig.add_trace(go.Bar(
                    name=param,
                    x=["Min", "Avg", "Max"],
                    y=[param_data["Min Values"], param_data["Avg Values"], param_data["Max Values"]],
                    text=[f"{param_data['Min Values']:.0f}", f"{param_data['Avg Values']:.1f}", f"{param_data['Max Values']:.0f}"],
                    textposition="auto"
                ))
            
            fig.update_layout(
                title="Parameter Variation Ranges Across All Microrobots",
                yaxis_title="Number of Values Tested",
                barmode="group"
            )
            
            st.plotly_chart(fig, use_container_width=True, key="param_ranges_chart")
        else:
            st.info("No parameter variation data available.")
    
    with tab3:
        st.subheader("Detailed Parameter Values")
        st.markdown("Explore the exact parameter values tested for each microrobot.")
        
        # Select microrobots to compare
        selected_microrobots = st.multiselect(
            "Select microrobots to analyze:",
            options=microrobots,
            default=microrobots[:min(3, len(microrobots))],
            key="detail_micro_select"
        )
        
        if not selected_microrobots:
            st.info("Please select at least one microrobot to analyze.")
        else:
            # Show detailed parameter values for each microrobot
            for robot in selected_microrobots:
                robot_data = all_data[all_data['Microrobot'] == robot]
                
                if not robot_data.empty:
                    st.markdown(f"#### {robot}")
                    
                    param_details = []
                    
                    for param in param_columns:
                        if param in robot_data.columns:
                            values = sorted([
                                str(val) for val in robot_data[param].unique() 
                                if pd.notna(val)
                            ])
                            
                            if values:
                                param_details.append({
                                    "Parameter": param,
                                    "Values Tested": ", ".join(values),
                                    "Number of Values": len(values)
                                })
                    
                    if param_details:
                        st.dataframe(pd.DataFrame(param_details), use_container_width=True)
                        
                        # Let user select a parameter to visualize
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            viz_param = st.selectbox(
                                "Select parameter:",
                                options=[p["Parameter"] for p in param_details],
                                key=f"param_select_{robot}"
                            )
                        
                        with col2:
                            if viz_param in robot_data.columns:
                                # Count occurrences of each value
                                value_counts = robot_data[viz_param].value_counts().reset_index()
                                value_counts.columns = [viz_param, 'Count']
                                
                                # Create bar chart
                                fig = px.bar(
                                    value_counts, 
                                    x=viz_param, 
                                    y='Count',
                                    title=f"Test Distribution for {viz_param}",
                                    color_discrete_sequence=["indigo"]
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"detail_chart_{robot}_{viz_param}")
    
    st.markdown("</div>", unsafe_allow_html=True)

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
    # Initialize saved plots in session state if not exists
    if 'saved_comparison_plots' not in st.session_state:
        st.session_state.saved_comparison_plots = []
    
    # Initialize form submission state if not exists
    if 'special_analysis_submitted' not in st.session_state:
        st.session_state.special_analysis_submitted = False
    
    # Initialize current params if not exists
    if 'current_analysis_params' not in st.session_state:
        st.session_state.current_analysis_params = {
            'x_param': None,
            'fixed_params': {},
        }
    
    # Initialize x_param in session state if not exists
    if 'x_param_value' not in st.session_state:
        st.session_state.x_param_value = None
    
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
    
    # Select X-axis parameter OUTSIDE the form to make it reactive
    available_params = list(param_values.keys())
    
    # Store previous value to detect changes
    prev_x_param = st.session_state.x_param_value
    
    # X-axis parameter selection
    x_param = st.selectbox("Select parameter for X-axis:", 
                          available_params, 
                          key="x_param_select")
    
    # Update session state
    st.session_state.x_param_value = x_param
    
    # If x_param has changed, reset fixed parameter selections for that param
    x_param_changed = prev_x_param != x_param and prev_x_param is not None
    
    # Create a form for the rest of the parameter selection
    with st.form(key="parameter_analysis_form"):
        # Create filters for fixed parameters
        st.subheader("Fixed Parameter Values")
        st.markdown("Select one value for each parameter (except the X-axis parameter):")
        
        fixed_params = {}
        col1, col2 = st.columns(2)
        
        # Filter out the current X-axis parameter
        left_params = [p for p in available_params if p != x_param][:len(available_params)//2]
        right_params = [p for p in available_params if p != x_param][len(available_params)//2:]
        
        with col1:
            for param in left_params:
                if param in param_values:
                    values = param_values[param]
                    if values:
                        # Ensure a default is selected if values exist
                        default_idx = 0 if values else None
                        selected = st.selectbox(f"{param}:", values, index=default_idx, key=f"fixed_{param}")
                        fixed_params[param] = [selected]
        
        with col2:
            for param in right_params:
                if param in param_values:
                    values = param_values[param]
                    if values:
                        # Ensure a default is selected if values exist
                        default_idx = 0 if values else None
                        selected = st.selectbox(f"{param}:", values, index=default_idx, key=f"fixed_{param}")
                        fixed_params[param] = [selected]
        
        # Submit button in the form
        submitted = st.form_submit_button("Analyze Parameter Effect")
        
        if submitted:
            # Store the analysis parameters in session state
            st.session_state.current_analysis_params = {
                'x_param': x_param,
                'fixed_params': fixed_params.copy(),
            }
            st.session_state.special_analysis_submitted = True
    
    # If x_param changed, we need to rerun to update the form options
    if x_param_changed:
        st.rerun()
    
    # Create a container for the plot
    plot_container = st.container()
    
    # Process the analysis if form was submitted (either now or previously)
    if st.session_state.special_analysis_submitted:
        # Get the parameters from session state
        x_param = st.session_state.current_analysis_params['x_param']
        fixed_params = st.session_state.current_analysis_params['fixed_params']
        
        # Filter data based on fixed parameters
        filtered_data = analysis_data.copy()
        
        for param, values in fixed_params.items():
            if param in filtered_data.columns and values:
                filtered_data = filtered_data[filtered_data[param].astype(str).isin([str(v) for v in values])]
        
        # Check for sufficient data
        if filtered_data.empty:
            with plot_container:
                st.error("No data matches the selected parameter values. Please adjust your selections.")
        
        # Display analysis if there's data for the selected parameter
        elif x_param in filtered_data.columns:
            # Calculate statistics
            grouped = filtered_data.groupby(x_param)['Head Deflection Angle [°]']
            stats_df = pd.DataFrame({
                'Mean Deflection': grouped.mean(),
                'Max Deflection': grouped.max(),
                'Min Deflection': grouped.min(),
            }).reset_index()
            
            # Add microrobot name to title if applicable
            microrobot_name = None
            title_suffix = ""
            if current_microrobot_data is not None and not current_microrobot_data.empty:
                microrobot_name = current_microrobot_data['Microrobot'].iloc[0]
                title_suffix = f" for {microrobot_name}"
            
            # Create plot title with parameter details
            plot_title = f"Variation of Deflection Angle with {x_param}{title_suffix}"
            
            # Create subplot labels
            subplot_params = [f"{param}: {values[0]}" for param, values in fixed_params.items()]
            subplot_label = " | ".join(subplot_params)
            
            # Create visualization
            fig = px.bar(
                stats_df, 
                x=x_param, 
                y='Mean Deflection',
                title=plot_title,
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
            
            # Layout improvements
            fig.update_layout(
                xaxis_title=x_param,
                yaxis_title='Deflection Angle [°]',
                legend_title="Statistics",
                barmode='group'
            )
            
            # Display current analysis
            with plot_container:
                st.subheader("Current Analysis")
                
                # Display fixed parameters
                with st.expander("Analysis Configuration", expanded=False):
                    fixed_params_df = pd.DataFrame({
                        "Parameter": list(fixed_params.keys()),
                        "Fixed Value": [v[0] for v in fixed_params.values()]
                    })
                    st.dataframe(fixed_params_df, use_container_width=True)
                
                st.plotly_chart(fig, use_container_width=True, key=f"special_analysis_{x_param}")
                
                # Save button
                plot_metadata = {
                    'id': str(uuid.uuid4()),  # Generate unique ID for this plot
                    'x_param': x_param,
                    'fixed_params': fixed_params.copy(),
                    'title': plot_title,
                    'subtitle': subplot_label,
                    'microrobot_name': microrobot_name,
                    'stats_df': stats_df.to_dict()
                }
                
                # Add separate "Add to comparison" button outside the form
                if st.button("Add to Customized Graphics", key="save_current_plot"):
                    st.session_state.saved_comparison_plots.append(plot_metadata)
                    st.success(f"✅ Plot add ! You now have {len(st.session_state.saved_comparison_plots)} graphics.")
                    st.rerun()
        else:
            with plot_container:
                st.error(f"Parameter {x_param} not found in the data.")
    
    # Display comparison section if we have saved plots
    if st.session_state.saved_comparison_plots:
        st.markdown("---")
        st.subheader("🔄 Customized Graphics")
        
        # Create column layout based on number of plots
        num_plots = len(st.session_state.saved_comparison_plots)
        cols_per_row = min(2, num_plots)  # Max 2 plots per row
        
        # Create comparison plots
        for i in range(0, num_plots, cols_per_row):
            plot_slice = st.session_state.saved_comparison_plots[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            
            for j, plot_data in enumerate(plot_slice):
                with cols[j]:
                    # Recreate plot from saved data
                    stats_df = pd.DataFrame.from_dict(plot_data['stats_df'])
                    fig = px.bar(
                        stats_df, 
                        x=plot_data['x_param'], 
                        y='Mean Deflection',
                        title=f"{plot_data['title']} (#{i+j+1})",
                        labels={'Mean Deflection': 'Mean Deflection Angle [°]'},
                        color_discrete_sequence=["royalblue"]
                    )
                    
                    # Add max markers
                    fig.add_scatter(
                        x=stats_df[plot_data['x_param']],
                        y=stats_df['Max Deflection'],
                        mode='markers',
                        name='Maximum',
                        marker=dict(color='red', size=10)
                    )
                    
                    # Layout improvements
                    fig.update_layout(
                        xaxis_title=plot_data['x_param'],
                        yaxis_title='Deflection Angle [°]',
                        legend_title="Statistics",
                        barmode='group',
                        height=400
                    )
                    
                    # Add subtitle with parameter details
                    st.caption(f"⚙️ Parameters: {plot_data['subtitle']}")
                    
                    # Plot chart
                    st.plotly_chart(fig, use_container_width=True, key=f"saved_plot_{plot_data['id']}")
                    
                    # Define a unique key for this remove button
                    remove_key = f"remove_{plot_data['id']}"
                    
                    # Remove button 
                    if st.button("🗑️ Remove", key=remove_key):
                        # Create a new list excluding the current plot
                        st.session_state.saved_comparison_plots = [
                            p for p in st.session_state.saved_comparison_plots if p['id'] != plot_data['id']
                        ]
                        st.success(f"Plot #{i+j+1} removed from graphic section.")
                        st.rerun()
        
        # Clear all button
        if st.button("🗑️ Clear All Graphics", key="clear_all_plots"):
            st.session_state.saved_comparison_plots = []
            st.success("All graphics plots cleared!")
            st.rerun()
    
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
    
    # Define common variables
    param_columns = ['Actuation Angle', 'Magnetic Distance [cm]', 'Gravity Force', 
                     'Actuation Mode', 'Flow Profile', 'Embedding length']
    
    # Initialize session state variables
    if 'show_microrobot_analysis' not in st.session_state:
        st.session_state.show_microrobot_analysis = True
    if 'show_global_analysis' not in st.session_state:
        st.session_state.show_global_analysis = True
    if 'show_special_analysis' not in st.session_state:
        st.session_state.show_special_analysis = True
    if 'show_unified_params' not in st.session_state:
        st.session_state.show_unified_params = False
    if 'saved_comparison_plots' not in st.session_state:
        st.session_state.saved_comparison_plots = []
    if 'show_parameter_variations' not in st.session_state:
        st.session_state.show_parameter_variations = True
    if 'show_all_variations' not in st.session_state:
        st.session_state.show_all_variations = True
        
    # Section toggles in sidebar
    st.sidebar.header("Dashboard Sections")

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
    st.session_state.show_unified_params = st.sidebar.toggle(
        "Show Unified Parameter Values", 
        value=st.session_state.show_unified_params
    )
    st.session_state.show_all_variations = st.sidebar.toggle(
        "Show All Microrobots Parameter Variations", 
        value=st.session_state.show_all_variations
    )
    
    # Load all data initially
    all_data, microrobots = load_all_files()
    param_values = get_parameter_values(all_data)
    
    # File uploader
    with st.container():
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 Upload an Excel file for analysis", type=["xlsx"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    # MOVED TO TOP: All microrobots parameter variations section
    if st.session_state.show_all_variations and all_data is not None:
        show_simple_parameter_variations_table(all_data)
    
    # Parameter filters section - Now with expander
    # The expander replaces the previous static header
    with st.expander("🔄 𝗨𝗻𝗶𝗳𝗶𝗲𝗱 𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿 𝗩𝗮𝗹𝘂𝗲𝘀", expanded=st.session_state.show_unified_params):
        st.markdown("<div class='section'>", unsafe_allow_html=True)
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
        
        # NEW FEATURE: Parameter Variations Table
        
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
                st.plotly_chart(fig1, use_container_width=True, key="distance_box")
        with col2:
            if 'Actuation Angle' in filtered_df.columns:
                fig2 = px.box(filtered_df, x='Actuation Angle', y='Head Deflection Angle [°]', 
                             title="Deflection vs Actuation Angle", color_discrete_sequence=["red"])
                st.plotly_chart(fig2, use_container_width=True, key="angle_box")
        
        col3, col4 = st.columns(2)
        with col3:
            if 'Flow Profile' in filtered_df.columns:
                fig3 = px.box(filtered_df, x='Flow Profile', y='Head Deflection Angle [°]', 
                             title="Deflection vs Flow Profile", color_discrete_sequence=["green"])
                st.plotly_chart(fig3, use_container_width=True, key="flow_box")
        with col4:
            if 'Actuation Mode' in filtered_df.columns:
                fig4 = px.box(filtered_df, x='Actuation Mode', y='Head Deflection Angle [°]', 
                             title="Deflection vs Actuation Mode", color_discrete_sequence=["purple"])
                st.plotly_chart(fig4, use_container_width=True, key="mode_box")
            
        col5, col6 = st.columns(2)
        with col5:
            if 'Gravity Force' in filtered_df.columns:
                fig5 = px.box(filtered_df, x='Gravity Force', y='Head Deflection Angle [°]', 
                             title="Deflection vs Gravity", color_discrete_sequence=["chocolate"])
                st.plotly_chart(fig5, use_container_width=True, key="gravity_box")
        with col6:
            if 'Embedding length' in filtered_df.columns:
                fig6 = px.box(filtered_df, x='Embedding length', y='Head Deflection Angle [°]', 
                             title="Deflection vs Embedding Length", color_discrete_sequence=["darkorange"])
                st.plotly_chart(fig6, use_container_width=True, key="embed_box")
        
        # Sensitivity pie chart
        sensitivity_data = calculate_sensitivity(filtered_df)
        sensitivity_df = pd.DataFrame(list(sensitivity_data.items()), columns=['Parameter', 'Sensitivity (%)'])
        fig7 = px.pie(sensitivity_df, values='Sensitivity (%)', names='Parameter', 
              title="Microrobot Deflection Sensitivity to Parameters", 
              color_discrete_sequence=["cyan", "magenta", "yellow", "orange", "gray"])
        st.plotly_chart(fig7, use_container_width=True, key="sensitivity_pie")
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
                st.plotly_chart(fig8, use_container_width=True, key="global_micro_box")
                
                max_deflection_by_micro = filtered_all_data.groupby('Microrobot')['Head Deflection Angle [°]'].max().reset_index()
                max_deflection_by_micro = max_deflection_by_micro.sort_values(by='Head Deflection Angle [°]', ascending=True).head(20)
                fig9 = px.bar(max_deflection_by_micro, x='Microrobot', y='Head Deflection Angle [°]', 
                             title="Top Microrobots by Max Deflection", color_discrete_sequence=["darkblue"])
                st.plotly_chart(fig9, use_container_width=True, key="global_micro_bar")
                
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
    
    # All microrobots parameter variations section (SIMPLIFIED VERSION)
    
    # Special parameter analysis section
    if st.session_state.show_special_analysis:
        # Pass current microrobot data if available
        current_microrobot_data = None
        if uploaded_file is not None:
            current_microrobot_data = filtered_df if 'filtered_df' in locals() else df
        
        perform_special_analysis(all_data, param_values, current_microrobot_data)
    
if __name__ == "__main__":
    main()
