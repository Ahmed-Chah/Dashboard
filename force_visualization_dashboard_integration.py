"""
Integration example for adding force visualization to the Microrobot Dashboard
-----------------------------------------------------------------------------
This module demonstrates how to integrate the force visualization component
into the Microrobot Analysis Dashboard.
"""

import streamlit as st
from force_visualization import create_force_visualization

def add_force_visualization_section():
    """
    Add force visualization section to the dashboard.
    This function should be called from the main dashboard code.
    """
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("🔄 Force Vector Visualization")
    st.markdown("""
    This section helps you understand how different forces (gravity, magnetic, drag) 
    are used in magneto-fluidic tests to deflect microrobot.
    """)
    
    # Simplified interface with only custom configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configure Visualization")
        
        # Custom parameters inputs
        custom_params = {}
        
        # Add gravity orientation selector
        gravity_orientation = st.radio(
            "Magnetic Force Direction:",
            options=["0° (Same as Gravity)", "90° (Perpendicular to Gravity)", "180° (Opposite to Gravity)"],
            index=0,
            key="gravity_orientation",
            horizontal=True
        )
        
        # Extract the angle value from the selection
        gravity_angle = int(gravity_orientation.split("°")[0])
        custom_params['gravity_angle'] = gravity_angle
        
        # Adjust actuator angle range based on gravity orientation
        min_angle = -90 if gravity_angle == 90 else (0 if gravity_angle == 180 else -90)
        max_angle = 90 if gravity_angle == 90 or gravity_angle == 180 else 0
        default_angle = 0 if gravity_angle == 0 or gravity_angle == 180 else 90 if gravity_angle == 90 else 0
        
        custom_params['angle_deg'] = st.slider(
            "Magnetic Angle (degrees):",
            min_value=float(min_angle),
            max_value=float(max_angle),
            value=float(default_angle),
            step=15.0,
            key="custom_angle"
        )
        
        # Distance parameter
        custom_params['distance'] = st.slider(
            "Magnetic Distance (cm):",
            min_value=13.0,
            max_value=30.0,
            value=20.0,
            step=0.5,
            key="custom_distance"
        )
        
        # Flow velocity parameter
        custom_params['flow_velocity'] = st.slider(
            "Flow Velocity (cm/s):",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.5,
            key="custom_flow"
        )
        
        # Add explanation about magnetic force direction
        st.info(f"""
        **Magnetic Force Direction {gravity_angle}°:**
        {
        "Magnetic force acts in the X,Z plane with angles from 0° to -90° (aligned with gravity)." if gravity_angle == 0 else
        "Magnetic force acts in the X,Y plane with angles from -90° to +90° (perpendicular to gravity)." if gravity_angle == 90 else
        "Magnetic force acts in the X,Z plane with angles from 0° to +90° (opposite to gravity)."
        }
        """)
    
    # Create and display the visualization
    with col2:
        st.subheader("3D Force Visualization")
        
        try:
            # Generate the visualization with custom parameters
            fig, description, _ = create_force_visualization('custom', custom_params)
            
            # Description
            st.markdown(f"**{description}**")
            
            # Display the 3D plot
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def add_integration_code_to_main():
    """
    Returns the code snippet to add to main dashboard for integration.
    """
    code = """
# Import the force visualization module
from force_visualization_dashboard_integration import add_force_visualization_section

# In the main function, add a new section for force visualization
if 'show_force_visualization' not in st.session_state:
    st.session_state.show_force_visualization = True

# Add toggle to sidebar
st.session_state.show_force_visualization = st.sidebar.toggle(
    "Show Force Visualization", 
    value=st.session_state.show_force_visualization
)

# Display force visualization section if enabled
if st.session_state.show_force_visualization:
    add_force_visualization_section()
"""
    return code


# This part is just for standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Force Visualization Test", layout="wide")
    st.title("Force Visualization Component Test")
    
    add_force_visualization_section()