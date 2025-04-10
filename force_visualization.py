"""
Force Visualization Module for Microrobot Dashboard
---------------------------------------------------
This module provides 3D force visualization capabilities for the Microrobot Analysis Dashboard.
It adapts the force calculations from Force_Torque_Interface.py and provides Plotly-based
3D visualizations that can be integrated into the Streamlit dashboard.
"""

import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------- Force Calculation Functions (from Force_Torque_Interface.py) -------------------

def grad(z):
    """Calculate magnetic field gradient at position z."""
    L, W, D = [0.1, 0.1, 0.04]
    br = 1.32
    gradB = br*((-2*L*W/(L**2 + W**2 + 4*z**2)**(3/2) - L*W/(2*z**2*np.sqrt(L**2 + W**2 + 4*z**2)))/
                (L**2*W**2/(4*z**2*(L**2 + W**2 + 4*z**2)) + 1)
                - (L*W*(-4*D - 4*z)/((2*D + 2*z)*(L**2 + W**2 + (D + z)*(4*D + 4*z))**(3/2))
                - 2*L*W/((2*D + 2*z)**2*np.sqrt(L**2 + W**2 + (D + z)*(4*D + 4*z))))/
                (L**2*W**2/((2*D + 2*z)**2*(L**2 + W**2 + (D + z)*(4*D + 4*z))) + 1)) / np.pi
    return gradB

def getMagneticForce(distance):
    """Calculate magnetic force at given distance (in cm)."""
    # Convert distance from cm to m
    distance_m = distance / 100.0
    diameterEXT, diameterInt, height = 2e-3, 1e-3, 1e-3
    volume = np.pi * (diameterEXT**2 - diameterInt**2) * height / 4
    remanentField = 1.01
    magnetization = remanentField / (4 * np.pi * 1e-7)
    const = magnetization * volume
    return np.abs(const * grad(distance_m))

def computeGravitationalForce():
    """Calculate gravitational force on the microrobot."""
    # Hardcoded density values
    rho_f = 1060  # Fluid density
    rho_m = 7200  # Magnet density
    rho_cut = 7850  # Cutting tool density
    g = 9.81  # Gravity constant
    
    diameterEXT, diameterInt, height = 2e-3, 1e-3, 1e-3
    Vm = np.pi * (diameterEXT**2 - diameterInt**2) * height / 4
    dext_cut, dint_cut, h_cut = 2.2e-3, 1.7e-3, 2.3e-3
    Vcut = np.pi * (dext_cut**2 - dint_cut**2) * h_cut / 4
    Vt = Vm + Vcut
    rho = (Vm / Vt) * rho_m + (1 - (Vm / Vt)) * rho_cut
    return np.abs(-Vt * (rho - rho_f) * g)

def computeDragForce(flowVelocityCmPerSec):
    """Calculate drag force based on flow velocity (in cm/s)."""
    # Hardcoded values
    rho_f = 1060  # Fluid density
    nu_f = 3.5e-3  # Fluid viscosity
    
    vm = 0
    vf = flowVelocityCmPerSec * 1e-2  # Convert to m/s
    diameterEXT, height = 2e-3, 1e-3
    r_t = diameterEXT / 2
    r_m = r_t
    r_cl = 0.640e-3 / 2
    alpha = 1 - (r_cl / r_t) ** 2
    Re = rho_f * (3e-3 - 2 * r_t) * (vm - vf) / nu_f
    if Re == 0:
        Re = 1e-5
    Cd = 24 / complex(Re) + 6 / (1 + np.sqrt(complex(Re))) + 0.4
    dragForce = np.abs(np.real(-0.5 * rho_f * alpha * np.pi * r_m**2 * Cd * (vm - vf)**2))
    return dragForce

def calculateBField(z):
    """Calculate magnetic field at position z (in cm)."""
    # Convert z from cm to m
    z_m = z / 100.0
    L, W, D = [0.1, 0.1, 0.04]
    br = 1.32
    return (br / np.pi) * (
        np.arctan((L * W) / (2 * z_m * np.sqrt(4 * z_m**2 + L**2 + W**2)))
        - np.arctan((L * W) / (2 * (D + z_m) * np.sqrt(4 * (D + z_m)**2 + L**2 + W**2)))
    )

def getMagneticTorque(distance, angle_deg):
    """Calculate magnetic torque at given distance (cm) and angle (degrees)."""
    # Convert distance from cm to m
    distance_m = distance / 100.0
    diameterEXT, diameterInt, height = 2e-3, 1e-3, 1e-3
    volume = np.pi * (diameterEXT**2 - diameterInt**2) * height / 4
    remanentField = 1.01
    magnetization = remanentField / (4 * np.pi * 1e-7)
    const = magnetization * volume
    Bfield = calculateBField(distance)
    angle_rad = math.radians(angle_deg)
    return const * Bfield * math.sin(angle_rad)

def calculate_deflection(distance, angle_deg, flow_velocity, gravity_angle=0, E_GEN4_V1=2.68e6):
    """
    Calculate deflection angle based on forces and material properties.
    
    Args:
        distance: Magnet distance in cm
        angle_deg: Magnetic field angle in degrees
        flow_velocity: Flow velocity in cm/s
        gravity_angle: Direction of magnetic force relative to gravity (0, 90, or 180 degrees)
        E_GEN4_V1: Young's modulus (Pa)
        
    Returns:
        Deflection angle in degrees and force components
    """
    angle_rad = math.radians(angle_deg)
    sign = 1 if angle_deg >= 0 else -1
    
    # Beam parameters
    r_ext = 1.71e-3
    r_int = 0.51e-3
    I = (np.pi / 4) * (r_ext**4 - r_int**4)
    
    # Calculate forces
    Fg = computeGravitationalForce()/2
    Fm = getMagneticForce(distance)
    Fd = float(np.squeeze(computeDragForce(flow_velocity)))
    
    # Gravity is always in the -Z direction
    Fg_vec = np.array([0, 0, -Fg])
    
    # Drag force is always along X-axis
    Fd_vec = np.array([Fd, 0, 0])
    
    # Magnetic force depends on orientation
    if gravity_angle == 0:
        # Magnetic force in X,Z plane (same direction as gravity)
        Fm_x = Fm * math.cos(angle_rad)
        Fm_z = Fm * math.sin(angle_rad)  # Will be negative for negative angles, aligned with gravity
        Fm_vec = np.array([Fm_x, 0, Fm_z])
    elif gravity_angle == 90:
        # Magnetic force in X,Y plane (perpendicular to gravity)
        Fm_x = Fm * math.cos(angle_rad)
        Fm_y = Fm * math.sin(angle_rad)
        Fm_vec = np.array([Fm_x, Fm_y, 0])
    elif gravity_angle == 180:
        # Magnetic force in X,Z plane (opposite to gravity)
        Fm_x = Fm * math.cos(angle_rad)
        Fm_z = Fm * math.sin(angle_rad)  # Positive to oppose gravity
        Fm_vec = np.array([Fm_x, 0, Fm_z])
    else:
        # Default (same as 0 degrees)
        Fm_x = Fm * math.cos(angle_rad)
        Fm_z = Fm * math.sin(angle_rad)
        Fm_vec = np.array([Fm_x, 0, Fm_z])
    
    # Magnetic torque
    Cm = np.array([0, getMagneticTorque(distance, angle_deg), 0])
    
    # Lever arm
    r = np.array([0.01, 0, 0])  # 10 mm
    
    # Calculate moment based on case
    if sign == 1:
        M = (np.cross(r, Fg_vec + Fd_vec + Fm_vec) + Cm) * 1.2
    else:
        M = (np.cross(r, Fg_vec + Fd_vec) + np.cross(r, Fm_vec) - Cm) * 1.2
    
    # Calculate moment magnitude and deflection
    M_norm = np.linalg.norm(M)
    theta_rad_deflection = M_norm / (E_GEN4_V1 * I)
    theta_deg_deflection = np.degrees(theta_rad_deflection)
    
    # Limit deflection to the actuation angle
    if abs(theta_deg_deflection) > abs(angle_deg):
        theta_deg_deflection = abs(angle_deg)
    
    # Determine deflection direction based on gravity orientation and forces
    if gravity_angle == 90:
        # For 90° (perpendicular), deflection sign should match actuator angle sign
        deflection_sign = np.sign(angle_deg)
    else:
        # For 0° and 180°, deflection direction depends on the specific configuration
        if gravity_angle == 0 and angle_deg < 0:
            # For 0° and negative angles, deflection should be downward
            deflection_sign = -1
        elif gravity_angle == 180 and angle_deg > 0:
            # For 180° and positive angles, deflection should be upward
            deflection_sign = 1
        else:
            # For other cases, use the sign of the moment
            deflection_sign = np.sign(M[1])
    
    # Apply the correct sign
    theta_deg_deflection = abs(theta_deg_deflection) * deflection_sign
    
    # Extract the components for visualization
    Fm_x = Fm_vec[0]
    Fm_y = Fm_vec[1]
    Fm_z = Fm_vec[2]
    
    # Calculate resultant force
    resultant = np.linalg.norm(Fg_vec + Fd_vec + Fm_vec)
    
    return {
        'theta_deg_deflection': theta_deg_deflection,
        'Fg': Fg,
        'Fm': Fm,
        'Fd': Fd,
        'Fm_x': Fm_x,
        'Fm_y': Fm_y,
        'Fm_z': Fm_z,
        'resultant': resultant,
        'M_norm': M_norm,
        'gravity_angle': gravity_angle
    }

def create_force_visualization(scenario, custom_params=None):
    """
    Create a 3D visualization of forces for a given scenario.
    
    Args:
        scenario: Should be 'custom' in this simplified version
        custom_params: Custom parameters (dict with keys: distance, angle_deg, flow_velocity, gravity_angle)
    
    Returns:
        Plotly figure object
    """
    # Default parameters
    distance = 20  # cm
    angle_deg = 0  # degrees
    flow_velocity = 0  # cm/s
    gravity_angle = 0  # degrees (magnetic force direction relative to gravity)
    
    # Use custom parameters
    if custom_params:
        distance = custom_params.get('distance', 20)
        angle_deg = custom_params.get('angle_deg', 0)
        flow_velocity = custom_params.get('flow_velocity', 0)
        gravity_angle = custom_params.get('gravity_angle', 0)
        title = f"Force Visualization (Distance: {distance}cm, Angle: {angle_deg}°, Flow: {flow_velocity}cm/s)"
        description = f"Magnetic force direction: {gravity_angle}°"
    else:
        title = "Force Visualization"
        description = "Default configuration"
    
    # Calculate deflection and forces
    result = calculate_deflection(distance, angle_deg, flow_velocity, gravity_angle)
    theta_deg_deflection = result['theta_deg_deflection']
    
    # Create the 3D visualization
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scene'}]],
        subplot_titles=[title]
    )
    
    # Parameters for visualization
    beam_length = 0.01  # 10 mm
    theta_rad_deflection = math.radians(abs(theta_deg_deflection))
    deflection_sign = np.sign(theta_deg_deflection)
    
    # Force vectors scaling
    max_force = max(result['Fg'], result['Fm'], result['Fd'])
    force_scale = beam_length * 0.5 / max_force if max_force > 0 else 1.0
    
    # Beam is always initially along x-axis
    x_undeflected = np.array([0, beam_length])
    y_undeflected = np.array([0, 0])
    z_undeflected = np.array([0, 0])
    
    # Calculate deflected beam position based on forces
    if gravity_angle == 90:
        # Deflection in horizontal plane (X,Y)
        x_deflected = np.array([0, beam_length * math.cos(theta_rad_deflection)])
        y_deflected = np.array([0, beam_length * math.sin(theta_rad_deflection) * deflection_sign])
        z_deflected = np.array([0, 0])
    else:
        # Deflection in vertical plane (X,Z)
        x_deflected = np.array([0, beam_length * math.cos(theta_rad_deflection)])
        y_deflected = np.array([0, 0])
        z_deflected = np.array([0, beam_length * math.sin(theta_rad_deflection) * deflection_sign])
    
    # Define beam endpoints
    beam_end = np.array([beam_length, 0, 0])
    deflected_end = np.array([x_deflected[1], y_deflected[1], z_deflected[1]])
    
    # Add beams to visualization
    fig.add_trace(
        go.Scatter3d(
            x=x_undeflected, y=y_undeflected, z=z_undeflected,
            mode='lines',
            line=dict(color='black', width=6),
            name='Initial Microrobot',
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=x_deflected, y=y_deflected, z=z_deflected,
            mode='lines',
            line=dict(color='black', width=6),
            name='Deflected Microrobot',
            showlegend=False
        )
    )
    
    # Add markers for beam endpoints
    # Undeflected position
    fig.add_trace(
        go.Scatter3d(
            x=[beam_end[0]], y=[beam_end[1]], z=[beam_end[2]],
            mode='markers',
            marker=dict(
                color='blue',
                size=10,
                symbol='circle'
            ),
            name='Initial Microrobot'
        )
    )
    
    # Deflected position
    fig.add_trace(
        go.Scatter3d(
            x=[deflected_end[0]], y=[deflected_end[1]], z=[deflected_end[2]],
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='circle'
            ),
            name='Deflected Microrobot'
        )
    )
    
    # Add force vectors
    # Gravity force - always downward
    if result['Fg'] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=[beam_end[0], beam_end[0]],
                y=[beam_end[1], beam_end[1]],
                z=[beam_end[2], beam_end[2] - force_scale * result['Fg']],
                mode='lines',
                line=dict(color='green', width=4),
                name='Fg - Gravity Force'
            )
        )
    
    # Drag force - always along x-axis
    if result['Fd'] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=[beam_end[0], beam_end[0] + force_scale * result['Fd']],
                y=[beam_end[1], beam_end[1]],
                z=[beam_end[2], beam_end[2]],
                mode='lines',
                line=dict(color='red', width=4),
                name='Fd - Drag Force'
            )
        )
    
    # Magnetic force - direction depends on gravity_angle
    if result['Fm'] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=[beam_end[0], beam_end[0] + force_scale * result['Fm_x']],
                y=[beam_end[1], beam_end[1] + force_scale * result['Fm_y']],
                z=[beam_end[2], beam_end[2] + force_scale * result['Fm_z']],
                mode='lines',
                line=dict(color='blue', width=4),
                name='Fm - Magnetic Force'
            )
        )
    
    # Configure the scene
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
        width=800,
        scene_dragmode='orbit'
    )
    
    return fig, description, {
        'Gravity Force': f"{result['Fg']:.2e} N",
        'Drag Force': f"{result['Fd']:.2e} N",
        'Magnetic Force': f"{result['Fm']:.2e} N",
        'Deflection Angle': f"{theta_deg_deflection:.2f}°",
        'Resultant Force': f"{result['resultant']:.2e} N",
        'Moment Magnitude': f"{result['M_norm']:.2e} N·m"
    }

# Predefined scenarios for the dashboard
SCENARIO_OPTIONS = [
    {'label': 'Magnetic Force (0° - Same as Gravity)', 'value': 'gravity_0'},
    {'label': 'Magnetic Force (90° - Perpendicular to Gravity)', 'value': 'gravity_90'},
    {'label': 'Magnetic Force (180° - Opposite to Gravity)', 'value': 'gravity_180'},
    {'label': 'Flow Effect - Drag Force', 'value': 'flow_effect'},
    {'label': 'Magnetic Distance Effect', 'value': 'magnetic_distance'},
    {'label': 'Custom Configuration', 'value': 'custom'}
]

# Scenario descriptions for tooltip or info section
SCENARIO_EXPLANATIONS = {
    'gravity_0': """
    In this scenario, the magnetic force is aligned with gravity (0°).
    The magnetic force acts in the X,Z plane with angles from 0° to -90°.
    This means the magnetic force can pull in the same general direction as gravity.
    Deflection occurs in the vertical plane (X,Z).
    """,
    
    'gravity_90': """
    Here, the magnetic force is perpendicular to gravity (90°).
    The magnetic force acts in the X,Y plane with angles from -90° to +90°.
    This means the magnetic force is acting in a horizontal plane perpendicular to gravity.
    Deflection occurs in the horizontal plane (X,Y).
    """,
    
    'gravity_180': """
    In this scenario, the magnetic force is opposing gravity (180°).
    The magnetic force acts in the X,Z plane with angles from 0° to +90°.
    This means the magnetic force can pull in the opposite direction from gravity.
    Deflection occurs in the vertical plane (X,Z).
    """,
    
    'flow_effect': """
    This scenario simulates fluid flow around the microrobot.
    The flow creates a drag force acting along the x-axis, which causes the beam to bend.
    The magnitude of the drag force depends on the flow velocity and fluid properties.
    """,
    
    'magnetic_distance': """
    This scenario demonstrates how the magnetic force changes with distance.
    A closer magnetic source (10cm instead of the standard 20cm) creates a stronger magnetic force.
    This shows how the distance parameter in experiments directly affects the force magnitude.
    """,
    
    'custom': """
    This custom scenario allows you to specify your own combination of parameters:
    - Magnetic Force Direction: Relation between magnetic force and gravity (0°, 90°, 180°)
    - Magnetic Angle: Specific angle within the respective plane
    - Distance: Distance between the microrobot and the magnetic source (cm)
    - Flow Velocity: Speed of fluid flow around the microrobot (cm/s)
    
    The actuator angle range automatically adjusts based on the selected magnetic force direction:
    - 0° (Same as Gravity): Actuator angle from 0° to -90° in X,Z plane
    - 90° (Perpendicular to Gravity): Actuator angle from -90° to +90° in X,Y plane
    - 180° (Opposite to Gravity): Actuator angle from 0° to +90° in X,Z plane
    """
}