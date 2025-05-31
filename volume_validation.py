import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Define the new base data directory
BASE_DATA_DIR = Path("/srv/azkaban_pipelines_migration_airflow-DEVELOP/utilities/schema_data_validation_and_comparison/dag_scripts/results")

def get_latest_dates(num_dates=5):
    dates = [d.name for d in BASE_DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(dates, reverse=True)[:num_dates]

def format_pipeline_name(filename):
    return filename.replace('alta_', 'Alta ').replace('_', ' ').title()

def get_all_pipelines():
    pipelines = set()
    for date_dir in BASE_DATA_DIR.iterdir():
        if date_dir.is_dir():
            row_counts_dir = date_dir / "row_counts"
            if row_counts_dir.exists():
                for file in row_counts_dir.glob("*.csv"):
                    pipeline = file.stem.split('_row_counts_comparison_')[0]
                    pipelines.add(pipeline)
    return sorted(pipelines)

def get_tables_for_pipeline(pipeline):
    tables = set()
    for date_str in get_latest_dates(5):
        csv_file = BASE_DATA_DIR / date_str / "row_counts" / f"{pipeline}_row_counts_comparison_{date_str}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            tables.update(df['TABLE_NAME'].unique())
    return sorted(tables)

def format_difference(value):
    if pd.isna(value):
        return ""
    if value > 0:
        return f"sf > {int(value)}"
    elif value < 0:
        return f"pg > {int(abs(value))}"
    return "0"

def calculate_abnormal_behavior_with_percentage(row):
    if pd.isna(row['ROW_COUNT_NS']) or pd.isna(row['ROW_COUNT_PG']) or row['ROW_COUNT_PG'] == 0:
        return "0"
    
    diff = abs(row['ROW_COUNT_NS'] - row['ROW_COUNT_PG'])
    percentage = (100*diff) / row['ROW_COUNT_PG']

    if percentage == 0:
        return "0"
    elif percentage > 2:
        return f'🔴 {percentage:.2f}%'
    elif percentage < 1:
        return f'🟢 {percentage:.2f}%'
    else:
        return f'🟡 {percentage:.2f}%'

def create_volume_table():
    st.title("New Schema vs Postgres Schema Row Difference")

    dates = get_latest_dates(5)
    if not dates:
        st.error("No data available")
        return

    pipelines = get_all_pipelines()
    if not pipelines:
        st.warning("No pipeline data found.")
        return

    pipeline_name_map = {p: format_pipeline_name(p) for p in pipelines}
    friendly_to_internal = {v: k for k, v in pipeline_name_map.items()}
    friendly_names = sorted(pipeline_name_map.values())

    # Pipeline selection
    current_selection = st.session_state.selected_pipeline
    if current_selection != "All" and current_selection not in pipeline_name_map:
        st.session_state.selected_pipeline = "All"
        current_selection = "All"

    selected_friendly = st.selectbox(
        "Select Pipeline Group", 
        ["All"] + friendly_names,
        index=0 if current_selection == "All" else friendly_names.index(pipeline_name_map[current_selection]) + 1,
        key="volume_pipeline_select"
    )

    # Update pipeline in session state
    new_pipeline = "All" if selected_friendly == "All" else friendly_to_internal[selected_friendly]
    if new_pipeline != st.session_state.selected_pipeline:
        st.session_state.selected_pipeline = new_pipeline
        st.session_state.selected_table = "All"

    # Get available tables
    if st.session_state.selected_pipeline == "All":
        all_tables = set()
        for pipeline in pipelines:
            all_tables.update(get_tables_for_pipeline(pipeline))
        available_tables = sorted(all_tables)
    else:
        available_tables = get_tables_for_pipeline(st.session_state.selected_pipeline)

    # Table selection
    selected_table = st.selectbox(
        "Select Table", 
        ["All"] + available_tables,
        index=0 if st.session_state.selected_table == "All" else available_tables.index(st.session_state.selected_table) + 1 if st.session_state.selected_table in available_tables else 0,
        key="volume_table_select"
    )
    st.session_state.selected_table = selected_table

    # Search button
    if st.button("Search", key="volume_search_button"):
        st.session_state.volume_data = None

    # Load data when search is clicked
    if st.session_state.get('volume_data') is None:
        selected_pipelines = pipelines if st.session_state.selected_pipeline == "All" else [st.session_state.selected_pipeline]
        all_data = []
        all_data_with_counts = []

        for pipeline in selected_pipelines:
            pretty_name = pipeline_name_map[pipeline]
            for date_str in dates:
                csv_file = BASE_DATA_DIR / date_str / "row_counts" / f"{pipeline}_row_counts_comparison_{date_str}.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    if 'ROW_COUNT_NS' in df.columns and 'ROW_COUNT_PG' in df.columns:
                        for _, row in df.iterrows():
                            if st.session_state.selected_table != "All" and row['TABLE_NAME'] != st.session_state.selected_table:
                                continue
                            difference = row['ROW_COUNT_NS'] - row['ROW_COUNT_PG']
                            all_data.append({
                                'Pipeline Name': pretty_name,
                                'Table Name': row['TABLE_NAME'],
                                'Date': datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d"),
                                'Difference': difference
                            })
                            all_data_with_counts.append({
                                'Pipeline Name': pretty_name,
                                'Table Name': row['TABLE_NAME'],
                                'Date': datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d"),
                                'ROW_COUNT_NS': row['ROW_COUNT_NS'],
                                'ROW_COUNT_PG': row['ROW_COUNT_PG'],
                                'Difference': difference
                            })

        if not all_data:
            st.warning("No volume validation data found for the selected filters.")
            return

        df = pd.DataFrame(all_data)
        pivot_df = df.pivot_table(
            index=['Pipeline Name', 'Table Name'],
            columns='Date',
            values='Difference',
            aggfunc='first'
        ).reset_index()

        formatted_dates = [datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates]
        
        for date in formatted_dates:
            if date not in pivot_df.columns:
                pivot_df[date] = float('nan')

        pivot_df = pivot_df[['Pipeline Name', 'Table Name'] + formatted_dates]
        for date_col in formatted_dates:
            pivot_df[date_col] = pivot_df[date_col].apply(format_difference)

        st.session_state.volume_data = pivot_df

        # Abnormal behavior table
        df_with_counts = pd.DataFrame(all_data_with_counts)
        abnormal_pivot = df_with_counts.pivot_table(
            index=['Pipeline Name', 'Table Name'],
            columns='Date',
            values=['ROW_COUNT_NS', 'ROW_COUNT_PG', 'Difference'],
            aggfunc='first'
        )

        abnormal_df = pd.DataFrame()
        for date in formatted_dates:
            if ('ROW_COUNT_NS', date) in abnormal_pivot.columns and ('ROW_COUNT_PG', date) in abnormal_pivot.columns:
                temp_df = abnormal_pivot[[('ROW_COUNT_NS', date), ('ROW_COUNT_PG', date)]].copy()
                temp_df.columns = ['ROW_COUNT_NS', 'ROW_COUNT_PG']
                temp_df['Abnormal'] = temp_df.apply(calculate_abnormal_behavior_with_percentage, axis=1)
                abnormal_df[date] = temp_df['Abnormal']

        abnormal_df = abnormal_df.reset_index()
        st.session_state.abnormal_df = abnormal_df

    if st.session_state.get('volume_data') is not None:
        st.dataframe(
            st.session_state.volume_data,
            use_container_width=True,
            column_config={
                "Pipeline Name": "Pipeline Name",
                "Table Name": "Table Name",
                **{date: {"name": date} for date in [datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates]}
            }
        )

        st.subheader("Abnormal Behavior Detection")
        
        # Add percentage threshold slider
        threshold = st.slider(
            "Filter by minimum percentage difference (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            key="volume_threshold"
        )

        if 'abnormal_df' in st.session_state:
            def extract_percentage(cell):
                if cell == '0':
                    return 0.0
                if isinstance(cell, str):
                    parts = cell.split()
                    if len(parts) >= 2 and parts[1].endswith('%'):
                        try:
                            return float(parts[1][:-1])
                        except:
                            return 0.0
                return 0.0

            date_columns = [col for col in st.session_state.abnormal_df.columns 
                          if col not in ['Pipeline Name', 'Table Name']]
            
            # Create a DataFrame with just the percentages using map() instead of applymap()
            percentage_df = st.session_state.abnormal_df[date_columns].map(extract_percentage)
            
            # Filter rows where any date column meets or exceeds the threshold
            mask = percentage_df.ge(threshold).any(axis=1)
            filtered_abnormal = st.session_state.abnormal_df[mask]

            st.dataframe(
                filtered_abnormal,
                use_container_width=True,
                column_config={
                    "Pipeline Name": "Pipeline Name",
                    "Table Name": "Table Name",
                    **{date: {"name": date} for date in date_columns}
                }
            )
            