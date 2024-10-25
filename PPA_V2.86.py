import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import asyncio
import os  # Import os module for file name manipulation
import time  # Import time module for timing

st.set_page_config(layout="wide")

# Place the logo and title side by side
col_logo, col_title = st.columns([1, 5])

'''with col_logo:
    st.image('/Users/coltonhoward/Desktop/Liberty Logo 2.png'
             , width=100)  # Replace with your logo path or URL'''

st.title('Liberty Energy Proppant Simulation - Unfinished')

# Sidebar with expanders
st.sidebar.title("Settings")

# 1. File Upload Section in an Expander
with st.sidebar.expander("1. File Upload", expanded=True):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file, skiprows=[1])  # Skip the second line (units)
    st.sidebar.success(f"File uploaded: {uploaded_file.name}")
    columns = data.columns.tolist()

    # Generate the output file name based on the uploaded file name
    base_name = os.path.splitext(uploaded_file.name)[0]
    output_filename = f"{base_name}_simulated_data.csv"

    # 2. Column Selection Section
    with st.sidebar.expander("2. CSV Channel Mapping", expanded=True):
        x_column = st.selectbox("Time ⤵️", columns, key='x_column')
        y1_column = st.selectbox("Design Prop Concentration ⤵️", columns, key='y1_column')
        y3_column = st.selectbox("Total Slurry Rate ⤵️", columns, key='y3_column')
        y4_column = st.selectbox("Pressure ⤵️", columns, key='y4_column')
        y5_column = st.selectbox("Total Proppant ⤵️", columns, key='y5_column')

    # Control Buttons in the main area
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        start_button = st.button("Start/Restart")
    with col2:
        pause_button = st.button("Pause")
    with col3:
        resume_button = st.button("Resume")
    with col4:
        recalculate_button = st.button("Calculate")
    with col5:
        defaults_button = st.button("Calculation Defaults")
    with col6:
        analysis_button = st.button("Analysis")

    # Manage State
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'last_fig' not in st.session_state:
        st.session_state.last_fig = None
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = False  # Flag to indicate if analysis mode is active
    if 'last_full_boxes_consumed_calc' not in st.session_state:
        st.session_state['last_full_boxes_consumed_calc'] = 0  # Track last full boxes consumed for calc prop

    # Initialize stored variables in session_state if not already there
    variables_to_initialize = [
        'x_full', 'y1_full', 'y3_full', 'y4_full', 'y5_full',
        'calc_ppa_ppr_full', 'calc_ppa_smooth_full', 'calc_clean_rate_full',
        'delta_t_full', 'incremental_clean_volume_full', 'total_calc_clean_volume_full',
        'incremental_proppant_full', 'calc_total_proppant_full'
    ]

    for var in variables_to_initialize:
        if var not in st.session_state:
            st.session_state[var] = pd.Series(dtype=float)

    # Handle the "Defaults" button click
    if defaults_button:
        # Reset calculation parameters to default values
        st.session_state['base_density'] = 8.33
        st.session_state['specific_gravity'] = 2.65
        st.session_state['ppr'] = 45
        st.session_state['pt_prop_factor'] = 1.0
        st.session_state['high_cal'] = 15.19
        st.session_state['low_cal'] = 8.33
        st.session_state['baby_beast'] = 1.0
        # No need to rerun the script; widgets will pick up the new values

    # 3. Simulation Parameters Section
    with st.sidebar.expander("3. Simulation Parameters", expanded=False):
        delay = st.number_input("Delay (ms):", min_value=100, value=st.session_state.get('delay', 1000), step=100, key='delay')
        index_increment = st.number_input("Index rows:", min_value=1, value=st.session_state.get('index_increment', 10), step=1, key='index_increment')
        smoothing_window = st.number_input("Prop Smooth:", min_value=1, value=st.session_state.get('smoothing_window', 10), step=1, key='smoothing_window')
        instructors_index = st.number_input("Instructors Index:", min_value=0.0, value=st.session_state.get('instructors_index', 1.0), step=0.01, key='instructors_index')

    # 4. Calculation Parameters Section
    with st.sidebar.expander("4. Calculation Parameters", expanded=False):
        base_density = st.number_input(
            "Base Density:",
            min_value=0.1,
            value=st.session_state.get('base_density', 8.33),
            key='base_density'
        )
        specific_gravity = st.number_input(
            "Sand SG:",
            min_value=0.1,
            value=st.session_state.get('specific_gravity', 2.65),
            key='specific_gravity'
        )
        ppr = st.number_input("PPR:", min_value=1, value=st.session_state.get('ppr', 45), key='ppr')
        pt_prop_factor = st.number_input("PT Factor:", min_value=0.1, value=st.session_state.get('pt_prop_factor', 1.0), key='pt_prop_factor')
        high_cal = st.number_input("High Cal:", min_value=0.1, value=st.session_state.get('high_cal', 15.19), key='high_cal')
        low_cal = st.number_input("Low Cal:", min_value=0.1, value=st.session_state.get('low_cal', 8.33), key='low_cal')
        baby_beast = st.number_input("Baby Beast Factor", min_value=0.1, value=st.session_state.get('baby_beast', 1.0), key='baby_beast')

    if start_button:
        st.session_state.running = True
        st.session_state.paused = False
        st.session_state.analysis_mode = False
        st.session_state.index = 0  # Start from the beginning
        # Reset stored variables
        for var in variables_to_initialize:
            st.session_state[var] = pd.Series(dtype=float)
        # Reset box tracking
        st.session_state['last_full_boxes_consumed_calc'] = 0

    if pause_button:
        st.session_state.paused = True
        st.session_state.running = False  # Stop the loop but keep data plotted
        st.session_state.analysis_mode = False

    if resume_button:
        if st.session_state.paused:
            st.session_state.running = True
            st.session_state.paused = False
            st.session_state.analysis_mode = False

    if recalculate_button:
        # Do not reset histories
        st.session_state.analysis_mode = False
        pass  # New parameters will be used in future calculations

    if analysis_button:
        st.session_state.running = False
        st.session_state.paused = True
        st.session_state.analysis_mode = True

    # Variables needed for calculations
    base_density = st.session_state.base_density
    specific_gravity = st.session_state.specific_gravity
    ppr = st.session_state.ppr
    pt_prop_factor = st.session_state.pt_prop_factor
    high_cal = st.session_state.high_cal
    low_cal = st.session_state.low_cal
    baby_beast = st.session_state.baby_beast
    delay = st.session_state.delay
    index_increment = st.session_state.index_increment
    smoothing_window = st.session_state.smoothing_window
    instructors_index = st.session_state.instructors_index  # Get the instructors index

    # Calculate maximum and minimum values for axes based on the entire dataset
    x_min = data[x_column].min()
    x_max = data[x_column].max() * 1.05
    y1_max = data[y1_column].max() * 1.5
    y3_max = data[y3_column].max() * 1.2
    y4_max = data[y4_column].max() * 1.05

    # Color dictionary
    y1_color = '#005903'  # green for design prop conc
    y2_color = '#17becf'  # teal (used for Calc Clean Rate)
    y3_color = '#0349fc'  # blue for slurry rate
    y4_color = '#ff0000'  # red for pressure
    calc_prop_color = '#FF5F1F'  # orange for calc prop
    total_prop_color = '#808080'  # grey, total prop color
    total_calc_prop_color = '#800080'  # purple, calc prop color
    delta_prop_color = '#9FE2BF'  # color for delta prop

    # Create placeholders for plot, numerical values, boxes, and analysis
    plot_placeholder = st.empty()
    numerical_values_placeholder = st.empty()
    boxes_placeholder_csv = st.empty()
    boxes_placeholder_calc = st.empty()
    analysis_placeholder = st.empty()  # Placeholder for analysis plots
    box_swap_placeholder = st.empty()  # Placeholder for "Box Swap" message

    # Function to perform calculations on new data increments
    def perform_calculations_on_new_data(x_new, y1_new, y3_new, y4_new, y5_new):
        # Calculations on new data
        # Use current parameters
        # Calculate avf
        avf = (1 / (8.33 * specific_gravity))
        # ppr_calc
        ppr_calc = ppr / 45
        # Slurry equation
        slurry = -0.000009 * y1_new ** 4 + 0.0007 * y1_new ** 3 - 0.0244 * y1_new ** 2 + 0.6125 * y1_new + 8.3362
        # PPA after shift
        ppa_shift = (slurry - base_density) / (1 - slurry * avf)
        # Delta PPA
        delta_ppa = y1_new - ppa_shift
        # Cal PPA
        low_point = (15.191 - high_cal)
        high_point = (15.191 + low_point - low_cal) / (1 - (15.191 + low_point) * avf)
        constant = (high_point - 0) / (88)
        calibrated_ppa = constant + constant * (y1_new - 0.25) / 0.25
        ppa_after_cal_shift = calibrated_ppa - delta_ppa
        # Calculated PPA PPR
        calc_ppa_ppr_new = ((ppa_after_cal_shift / ppr_calc) * baby_beast) / pt_prop_factor
        # Smooth calc_ppa_ppr_new
        calc_ppa_smooth_new = calc_ppa_ppr_new.rolling(
            window=int(smoothing_window), center=True, min_periods=1
        ).mean().round(2)
        # Calculate CFR and Calculated Clean Rate
        ppa_new = calc_ppa_ppr_new  # PPA
        cfr_new = 1 / (ppa_new * avf + 1)  # CFR
        calc_clean_rate_new = y3_new * cfr_new  # Calculated Clean Rate
        # Compute delta_t
        delta_t_new = x_new.diff().fillna(0)
        # delta_t_new = delta_t_new / 60  # Uncomment if time is in seconds and needs conversion
        # Compute incremental clean volume (bbl)
        incremental_clean_volume_new = calc_clean_rate_new * delta_t_new  # bpm * min = bbl
        # Compute cumulative clean volume
        if not st.session_state.total_calc_clean_volume_full.empty:
            total_calc_clean_volume_new = st.session_state.total_calc_clean_volume_full.iloc[-1] + incremental_clean_volume_new.cumsum()
        else:
            total_calc_clean_volume_new = incremental_clean_volume_new.cumsum()
        # Compute incremental proppant (lb)
        incremental_proppant_new = incremental_clean_volume_new * 42 * ppa_new  # bbl * gal/bbl * lb/gal = lb
        # Compute cumulative total proppant
        if not st.session_state.calc_total_proppant_full.empty:
            calc_total_proppant_new = st.session_state.calc_total_proppant_full.iloc[-1] + incremental_proppant_new.cumsum()
        else:
            calc_total_proppant_new = incremental_proppant_new.cumsum()
        # Append new data to stored variables
        st.session_state.x_full = pd.concat([st.session_state.x_full, x_new], ignore_index=True)
        st.session_state.y1_full = pd.concat([st.session_state.y1_full, y1_new], ignore_index=True)
        st.session_state.y3_full = pd.concat([st.session_state.y3_full, y3_new], ignore_index=True)
        st.session_state.y4_full = pd.concat([st.session_state.y4_full, y4_new], ignore_index=True)
        st.session_state.y5_full = pd.concat([st.session_state.y5_full, y5_new], ignore_index=True)
        st.session_state.calc_ppa_ppr_full = pd.concat([st.session_state.calc_ppa_ppr_full, calc_ppa_ppr_new], ignore_index=True)
        st.session_state.calc_ppa_smooth_full = pd.concat([st.session_state.calc_ppa_smooth_full, calc_ppa_smooth_new], ignore_index=True)
        st.session_state.calc_clean_rate_full = pd.concat([st.session_state.calc_clean_rate_full, calc_clean_rate_new], ignore_index=True)
        st.session_state.delta_t_full = pd.concat([st.session_state.delta_t_full, delta_t_new], ignore_index=True)
        st.session_state.incremental_clean_volume_full = pd.concat([st.session_state.incremental_clean_volume_full, incremental_clean_volume_new], ignore_index=True)
        st.session_state.total_calc_clean_volume_full = pd.concat([st.session_state.total_calc_clean_volume_full, total_calc_clean_volume_new], ignore_index=True)
        st.session_state.incremental_proppant_full = pd.concat([st.session_state.incremental_proppant_full, incremental_proppant_new], ignore_index=True)
        st.session_state.calc_total_proppant_full = pd.concat([st.session_state.calc_total_proppant_full, calc_total_proppant_new], ignore_index=True)
        # Return the current total proppant
        current_calc_total_proppant = st.session_state.calc_total_proppant_full.iloc[-1]
        return current_calc_total_proppant

    # Function to display boxes with gradient fill to simulate emptying
    def display_boxes(boxes_consumed, total_boxes, num_boxes_to_display, label):
        st.write(f"**{label} Boxes**")
        total_boxes = int(total_boxes)
        num_boxes_to_display = int(num_boxes_to_display)
        if total_boxes <= 0:
            st.write("No boxes to display.")
            return
        # Ensure boxes_per_box_display is 1 (full boxes)
        boxes_per_box_display = 1
        cols = st.columns(num_boxes_to_display)
        for i in range(num_boxes_to_display):
            # Each display box represents one actual box
            box_label = f"{i + 1}"
            # Calculate the amount consumed in this box
            start_capacity = i * 25000
            end_capacity = (i + 1) * 25000
            consumed_in_box = min(max(boxes_consumed * 25000 - start_capacity, 0), 25000)
            fill_percentage = (consumed_in_box / 25000) * 100  # Percentage of box consumed

            # Create the box with gradient fill to simulate emptying
            box_html = f'''
                <div style="text-align:center;">
                    <div style="position: relative; width: 30px; height: 60px; border:1px solid black; background-color: #EE2827;">
                        <div style="
                            position: absolute;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: {fill_percentage}%;
                            background-color: #262626; 
                        "></div>
                    </div>
                    <div style="font-size:10px;">{box_label}</div>
                </div>
            '''
            cols[i].markdown(box_html, unsafe_allow_html=True)

    # Asynchronous function to display "Box Swap" message
    async def show_box_swap_message():
        box_swap_placeholder.markdown("<h2 style='text-align: center; color: red;'>Box Swap</h2>", unsafe_allow_html=True)
        await asyncio.sleep(2)
        box_swap_placeholder.empty()

    # Asynchronous function to update the plot
    async def update_plot():
        while st.session_state.running and st.session_state.index < len(data):
            # Determine start and end indices for the new data increment
            start_index = st.session_state.index
            end_index = st.session_state.index + index_increment

            # Ensure end_index does not exceed data length
            end_index = min(end_index, len(data))

            # Extract new data slices
            x_new = data[x_column].iloc[start_index:end_index].reset_index(drop=True)
            y1_new = data[y1_column].iloc[start_index:end_index].reset_index(drop=True)
            y3_new = data[y3_column].iloc[start_index:end_index].reset_index(drop=True)
            y4_new = data[y4_column].iloc[start_index:end_index].reset_index(drop=True)
            y5_new = data[y5_column].iloc[start_index:end_index].reset_index(drop=True) * instructors_index  # Apply instructors index

            # Perform calculations on new data and update stored variables
            current_calc_total_proppant = perform_calculations_on_new_data(x_new, y1_new, y3_new, y4_new, y5_new)

            # Update index
            st.session_state.index = end_index

            # Create the plot using stored data
            fig = go.Figure()

            # Add traces
            fig.add_trace(go.Scatter(
                x=st.session_state.x_full,
                y=st.session_state.y1_full,
                name=y1_column,
                line=dict(color=y1_color),
                yaxis='y1'
            ))

            fig.add_trace(go.Scatter(
                x=st.session_state.x_full,
                y=st.session_state.calc_ppa_smooth_full,
                name='Calc Prop Conc',
                line=dict(color=calc_prop_color),
                yaxis='y1',
                hovertemplate='%{y:.2f}'
            ))

            fig.add_trace(go.Scatter(
                x=st.session_state.x_full,
                y=st.session_state.calc_clean_rate_full,
                name='Calc Clean Rate',
                line=dict(color=y2_color),
                yaxis='y3'
            ))

            fig.add_trace(go.Scatter(
                x=st.session_state.x_full,
                y=st.session_state.y3_full,
                name=y3_column,
                line=dict(color=y3_color),
                yaxis='y3'
            ))

            fig.add_trace(go.Scatter(
                x=st.session_state.x_full,
                y=st.session_state.y4_full,
                name=y4_column,
                line=dict(color=y4_color),
                yaxis='y4'
            ))

            # Update layout
            fig.update_layout(
                xaxis=dict(
                    domain=[0.05, 0.95],
                    range=[x_min, x_max],
                ),
                yaxis=dict(
                    title=y1_column,
                    titlefont=dict(color=y1_color),
                    tickfont=dict(color=y1_color),
                    range=[0, y1_max],
                    showgrid=True,
                ),
                yaxis3=dict(
                    title="Rate (bpm)",
                    titlefont=dict(color=y3_color),
                    tickfont=dict(color=y3_color),
                    anchor='free',
                    overlaying='y',
                    side='right',
                    position=0.9,
                    range=[0, y3_max],
                    showgrid=False,
                ),
                yaxis4=dict(
                    title=y4_column,
                    titlefont=dict(color=y4_color),
                    tickfont=dict(color=y4_color),
                    anchor='free',
                    overlaying='y',
                    side='right',
                    position=0.95,
                    range=[0, y4_max],
                    showgrid=False,
                ),
                legend=dict(
                    x=0.5,
                    y=1.15,
                    xanchor='center',
                    orientation='h'
                ),
                margin=dict(l=0, r=0, t=30, b=10),
                autosize=True,
            )

            # Display the plot with container width to dynamically adjust
            plot_placeholder.plotly_chart(fig, use_container_width=True)

            # Store the last plot in session state
            st.session_state.last_fig = fig

            # Update numerical values
            # Get current values
            current_y1_value = st.session_state.y1_full.iloc[-1]
            current_calc_ppa_smooth_value = st.session_state.calc_ppa_smooth_full.iloc[-1]
            current_calc_clean_rate_value = st.session_state.calc_clean_rate_full.iloc[-1]
            current_y3_value = st.session_state.y3_full.iloc[-1]
            current_y4_value = st.session_state.y4_full.iloc[-1]
            current_y5_value = st.session_state.y5_full.iloc[-1]
            current_total_proppant_csv = st.session_state.y5_full.iloc[-1]

            # Calculate the proppant difference
            prop_diff = current_calc_total_proppant - current_total_proppant_csv

            # Update the numerical values placeholder
            with numerical_values_placeholder.container():
                cols = st.columns(8)
                # Define a function to create a colored metric
                def colored_metric(label, value, color):
                    return f"""
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 16px; color: {color};">{label}</p>
                        <p style="margin: 0; font-size: 24px; color: {color}; font-weight: bold;">{value}</p>
                    </div>
                    """
                cols[0].markdown(colored_metric("Design Prop Conc", f"{current_y1_value:.2f}", "green"), unsafe_allow_html=True)
                cols[1].markdown(colored_metric("Calc Prop Conc", f"{current_calc_ppa_smooth_value:.2f}", "orange"), unsafe_allow_html=True)
                cols[2].markdown(colored_metric("Calc Clean Rate", f"{current_calc_clean_rate_value:.2f}", "#17becf"), unsafe_allow_html=True)
                cols[3].markdown(colored_metric(y3_column, f"{current_y3_value:.2f}", "blue"), unsafe_allow_html=True)
                cols[4].markdown(colored_metric(y4_column, f"{current_y4_value:.0f}", "red"), unsafe_allow_html=True)
                cols[5].markdown(colored_metric(y5_column, f"{current_y5_value:.0f}", "#808080"), unsafe_allow_html=True)
                cols[6].markdown(colored_metric("Calc Total Prop (lbs)", f"{current_calc_total_proppant:,.0f}", "#800080"), unsafe_allow_html=True)
                cols[7].markdown(colored_metric("Calc - Total Prop Delta (lbs)", f"{prop_diff:,.0f}", "#9FE2BF"), unsafe_allow_html=True)

            # Update the boxes placeholders
            with boxes_placeholder_csv.container():
                # Compute total boxes and boxes consumed for CSV data
                total_proppant_max_csv = (data[y5_column] * instructors_index).max()  # Apply instructors index
                total_boxes_csv = int(np.ceil(total_proppant_max_csv / 25000))
                total_boxes_csv = max(1, total_boxes_csv)  # Ensure at least one box
                current_total_proppant_csv = st.session_state.y5_full.iloc[-1]
                boxes_consumed_csv = current_total_proppant_csv / 25000
                # Determine the number of boxes to display (full boxes only)
                num_boxes_to_display_csv = min(total_boxes_csv, 30)

                display_boxes(boxes_consumed_csv, total_boxes_csv, num_boxes_to_display_csv, label="Total Proppant (CSV)")

            with boxes_placeholder_calc.container():
                # Compute total boxes and boxes consumed for calculated data
                total_proppant_max_calc = st.session_state.calc_total_proppant_full.max()
                total_boxes_calc = int(np.ceil(total_proppant_max_calc / 25000))
                total_boxes_calc = max(1, total_boxes_calc)  # Ensure at least one box
                current_total_proppant_calc = st.session_state.calc_total_proppant_full.iloc[-1]
                boxes_consumed_calc = current_total_proppant_calc / 25000
                num_boxes_to_display_calc = max(num_boxes_to_display_csv, min(total_boxes_calc, 30))

                display_boxes(boxes_consumed_calc, total_boxes_calc, num_boxes_to_display_calc, label="Calculated Total Proppant")

            # Calculate number of fully consumed boxes
            full_boxes_consumed_calc = int(boxes_consumed_calc)

            # Check for box swap
            if full_boxes_consumed_calc > st.session_state['last_full_boxes_consumed_calc']:
                # Update the last_full_boxes_consumed_calc
                st.session_state['last_full_boxes_consumed_calc'] = full_boxes_consumed_calc
                # Schedule the box swap message
                asyncio.create_task(show_box_swap_message())

            # Wait for the specified delay
            await asyncio.sleep(delay / 1000)

            # Check if paused or analysis mode is activated
            if st.session_state.paused or st.session_state.analysis_mode:
                break

        if st.session_state.index >= len(data):
            st.session_state.running = False

    # Run the asynchronous function if the simulation is running
    if st.session_state.running:
        asyncio.run(update_plot())
    else:
        # Display the last plot if available
        if st.session_state.last_fig is not None:
            plot_placeholder.plotly_chart(st.session_state.last_fig, use_container_width=True)
            # Update numerical values
            current_y1_value = st.session_state.y1_full.iloc[-1]
            current_calc_ppa_smooth_value = st.session_state.calc_ppa_smooth_full.iloc[-1]
            current_calc_clean_rate_value = st.session_state.calc_clean_rate_full.iloc[-1]
            current_y3_value = st.session_state.y3_full.iloc[-1]
            current_y4_value = st.session_state.y4_full.iloc[-1]
            current_y5_value = st.session_state.y5_full.iloc[-1]
            current_calc_total_proppant = st.session_state.calc_total_proppant_full.iloc[-1]
            current_total_proppant_csv = st.session_state.y5_full.iloc[-1]

            # Calculate the proppant difference
            prop_diff = current_calc_total_proppant - current_total_proppant_csv

            # Update numerical values placeholder
            with numerical_values_placeholder.container():
                cols = st.columns(8)
                # Define the colored_metric function
                def colored_metric(label, value, color):
                    return f"""
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 16px; color: {color};">{label}</p>
                        <p style="margin: 0; font-size: 24px; color: {color}; font-weight: bold;">{value}</p>
                    </div>
                    """
                cols[0].markdown(colored_metric("Design Prop Conc", f"{current_y1_value:.2f}", "green"), unsafe_allow_html=True)
                cols[1].markdown(colored_metric("Calc Prop Conc", f"{current_calc_ppa_smooth_value:.2f}", "orange"), unsafe_allow_html=True)
                cols[2].markdown(colored_metric("Calc Clean Rate", f"{current_calc_clean_rate_value:.2f}", "#17becf"), unsafe_allow_html=True)
                cols[3].markdown(colored_metric(y3_column, f"{current_y3_value:.2f}", "blue"), unsafe_allow_html=True)
                cols[4].markdown(colored_metric(y4_column, f"{current_y4_value:.0f}", "red"), unsafe_allow_html=True)
                cols[5].markdown(colored_metric(y5_column, f"{current_y5_value:.0f}", "#808080"), unsafe_allow_html=True)
                cols[6].markdown(colored_metric("Calc Total Prop (lbs)", f"{current_calc_total_proppant:,.0f}", "#800080"), unsafe_allow_html=True)
                cols[7].markdown(colored_metric("Calc - Total Prop Delta (lbs)", f"{prop_diff:,.0f}", "#9FE2BF"), unsafe_allow_html=True)

            # Update the boxes placeholders
            with boxes_placeholder_csv.container():
                total_proppant_max_csv = (data[y5_column] * instructors_index).max()  # Apply instructors index
                total_boxes_csv = int(np.ceil(total_proppant_max_csv / 25000))
                total_boxes_csv = max(1, total_boxes_csv)  # Ensure at least one box
                current_total_proppant_csv = st.session_state.y5_full.iloc[-1]
                boxes_consumed_csv = current_total_proppant_csv / 25000
                num_boxes_to_display_csv = min(total_boxes_csv, 30)

                display_boxes(boxes_consumed_csv, total_boxes_csv, num_boxes_to_display_csv, label="Total Proppant (CSV)")

            with boxes_placeholder_calc.container():
                total_proppant_max_calc = st.session_state.calc_total_proppant_full.max()
                total_boxes_calc = int(np.ceil(total_proppant_max_calc / 25000))
                total_boxes_calc = max(1, total_boxes_calc)  # Ensure at least one box
                current_total_proppant_calc = st.session_state.calc_total_proppant_full.iloc[-1]
                boxes_consumed_calc = current_total_proppant_calc / 25000
                num_boxes_to_display_calc = max(num_boxes_to_display_csv, min(total_boxes_calc, 30))

                display_boxes(boxes_consumed_calc, total_boxes_calc, num_boxes_to_display_calc, label="Calculated Total Proppant")

            # Display analysis plots if in analysis mode
            if st.session_state.analysis_mode:
                with analysis_placeholder.container():
                    st.header("Data Analysis")

                    # Plot the difference between Total Prop and Calculated Total Prop
                    prop_diff_series = st.session_state.calc_total_proppant_full - st.session_state.y5_full
                    fig_diff = go.Figure()
                    fig_diff.add_trace(go.Scatter(
                        x=st.session_state.x_full,
                        y=prop_diff_series,
                        name='Prop Difference',
                        line=dict(color='#9FE2BF')
                    ))
                    fig_diff.update_layout(
                        title='Difference Between Calculated Total Prop and Total Prop',
                        xaxis_title='Time',
                        yaxis_title='Difference (lbs)',
                        autosize=True,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    st.plotly_chart(fig_diff, use_container_width=True)

                    # Plot Time vs Total Prop and Calculated Total Prop
                    fig_total_prop = go.Figure()
                    fig_total_prop.add_trace(go.Scatter(
                        x=st.session_state.x_full,
                        y=st.session_state.y5_full,
                        name='Total Prop',
                        line=dict(color='#808080')
                    ))
                    fig_total_prop.add_trace(go.Scatter(
                        x=st.session_state.x_full,
                        y=st.session_state.calc_total_proppant_full,
                        name='Calculated Total Prop',
                        line=dict(color='#800080')
                    ))
                    fig_total_prop.update_layout(
                        title='Time vs Total Prop and Calculated Total Prop',
                        xaxis_title='Time',
                        yaxis_title='Proppant (lbs)',
                        autosize=True,
                        margin=dict(l=40, r=40, t=80, b=40),  # Increased top margin for legend
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.10,
                            xanchor='center',
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig_total_prop, use_container_width=True)

                    # Plot Time vs Prop Conc and Calculated Prop Conc
                    fig_prop_conc = go.Figure()
                    fig_prop_conc.add_trace(go.Scatter(
                        x=st.session_state.x_full,
                        y=st.session_state.y1_full,
                        name='Design Prop Conc',
                        line=dict(color='#005903')
                    ))
                    fig_prop_conc.add_trace(go.Scatter(
                        x=st.session_state.x_full,
                        y=st.session_state.calc_ppa_smooth_full,
                        name='Calculated Prop Conc',
                        line=dict(color='#FF5F1F')
                    ))
                    fig_prop_conc.update_layout(
                        title='Time vs Prop Conc and Calculated Prop Conc',
                        xaxis_title='Time',
                        yaxis_title='Concentration',
                        autosize=True,
                        margin=dict(l=40, r=40, t=80, b=40),  # Increased top margin for legend
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.10,
                            xanchor='center',
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig_prop_conc, use_container_width=True)

            else:
                analysis_placeholder.empty()  # Clear the analysis placeholder if not in analysis mode

            # Create a DataFrame for export with additional variables
            export_data = pd.DataFrame({
                'Time': st.session_state.x_full,
                'Design Prop Conc': st.session_state.y1_full,
                'Calc Prop Conc': st.session_state.calc_ppa_smooth_full,
                'Calc Clean Rate': st.session_state.calc_clean_rate_full,
                'Total Slurry Rate': st.session_state.y3_full,
                'Pressure': st.session_state.y4_full,
                'Total Prop': st.session_state.y5_full,
                'Calc Total Prop': st.session_state.calc_total_proppant_full,
                'Prop Difference': st.session_state.calc_total_proppant_full - st.session_state.y5_full,
                'delta_t': st.session_state.delta_t_full,
                'Incremental Clean Volume': st.session_state.incremental_clean_volume_full,
                'Total Clean Volume': st.session_state.total_calc_clean_volume_full,
                'Incremental Proppant': st.session_state.incremental_proppant_full,
                # Include any other data you wish to export
            })
            # Add a download button with the generated file name
            csv = export_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=output_filename,  # Use the generated file name here
                mime='text/csv',
            )
        else:
            st.write("Please start the simulation to see the plot and numerical values.")

else:
    st.write("Please upload a CSV file from the sidebar to begin.")
