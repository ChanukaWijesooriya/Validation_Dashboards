import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, date

def get_latest_dates(num_dates=5):
    data_dir = Path("data")
    dates = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(dates, reverse=True)[:num_dates]

def format_pipeline_name(filename):
    return filename.replace('alta_', 'Alta ').replace('_', ' ').title()

def get_all_pipelines():
    pipelines = set()
    for date_dir in Path("data").iterdir():
        if date_dir.is_dir():
            row_counts_dir = date_dir / "partition_row_counts"
            if row_counts_dir.exists():
                for file in row_counts_dir.glob("*.csv"):
                    pipeline = file.stem.split('_partition_row_counts_comparison_')[0]
                    pipelines.add(pipeline)
    return sorted(pipelines)

def get_tables_for_pipeline(pipeline):
    tables = set()
    for date_str in get_latest_dates(5):
        csv_file = Path("data") / date_str / "partition_row_counts" / f"{pipeline}_partition_row_counts_comparison_{date_str}.csv"
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

def calculate_abnormal_behavior_partition(row):
    if pd.isna(row['ROW_COUNT_NS']) or pd.isna(row['ROW_COUNT_PG']) or row['ROW_COUNT_PG'] == 0:
        return "0"
    diff = abs(row['ROW_COUNT_NS'] - row['ROW_COUNT_PG'])
    percentage = (100 * diff) / row['ROW_COUNT_PG']
    if percentage == 0:
        return "0"
    elif percentage > 2:
        return f"🔴 {percentage:.2f}%"
    elif percentage < 1:
        return f"🟢 {percentage:.2f}%"
    else:
        return f"🟡 {percentage:.2f}%"

def create_partition_table():
    st.title("New Schema vs Postgres Schema Partition Row Count Difference")

    dates = get_latest_dates(5)
    pipelines = get_all_pipelines()
    pipeline_name_map = {p: format_pipeline_name(p) for p in pipelines}
    friendly_to_internal = {v: k for k, v in pipeline_name_map.items()}
    friendly_names = sorted(pipeline_name_map.values())

    current_selection = st.session_state.selected_pipeline
    if current_selection != "All" and current_selection not in pipeline_name_map:
        st.session_state.selected_pipeline = "All"
        current_selection = "All"

    col1, col2 = st.columns(2)
    with col1:
        selected_friendly = st.selectbox(
            "Select Pipeline Group",
            ["All"] + friendly_names,
            index=0 if current_selection == "All" else friendly_names.index(pipeline_name_map[current_selection]) + 1,
            key="partition_pipeline_select"
        )

    new_pipeline = "All" if selected_friendly == "All" else friendly_to_internal[selected_friendly]
    if new_pipeline != st.session_state.selected_pipeline:
        st.session_state.selected_pipeline = new_pipeline
        st.session_state.selected_table = "All"

    if st.session_state.selected_pipeline == "All":
        all_tables = set()
        for pipeline in pipelines:
            all_tables.update(get_tables_for_pipeline(pipeline))
        available_tables = sorted(all_tables)
    else:
        available_tables = get_tables_for_pipeline(st.session_state.selected_pipeline)

    with col2:
        selected_table = st.selectbox(
            "Select Table",
            ["All"] + available_tables,
            index=0 if st.session_state.selected_table == "All" else available_tables.index(st.session_state.selected_table) + 1 if st.session_state.selected_table in available_tables else 0,
            key="partition_table_select"
        )
    st.session_state.selected_table = selected_table

    if st.button("Search", key="partition_search_button"):
        st.session_state.partition_data = None
        st.session_state.partition_raw = None
        st.session_state.abnormal_df = None

    if st.session_state.get('partition_raw') is None:
        all_data = []
        all_data_with_counts = []
        all_partition_dates = set()

        selected_pipelines = [st.session_state.selected_pipeline] if st.session_state.selected_pipeline != "All" else pipelines
        for pipeline in selected_pipelines:
            pretty_name = pipeline_name_map[pipeline]
            for date_str in dates:
                csv_file = Path("data") / date_str / "partition_row_counts" / f"{pipeline}_partition_row_counts_comparison_{date_str}.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    if 'ROW_COUNT_NS' in df.columns and 'ROW_COUNT_PG' in df.columns:
                        for _, row in df.iterrows():
                            if st.session_state.selected_table != "All" and row['TABLE_NAME'] != st.session_state.selected_table:
                                continue
                            try:
                                partition_date = pd.to_datetime(row['PARTITION_COLUMN_VALUE']).date()
                                all_partition_dates.add(partition_date)
                                all_data.append({
                                    'Pipeline Name': pretty_name,
                                    'Table Name': row['TABLE_NAME'],
                                    'Partition': row['PARTITION_COLUMN_VALUE'],
                                    'PartitionDate': partition_date,
                                    'Date': datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d"),
                                    'Difference': row['ROW_COUNT_NS'] - row['ROW_COUNT_PG']
                                })
                                all_data_with_counts.append({
                                    'Pipeline Name': pretty_name,
                                    'Table Name': row['TABLE_NAME'],
                                    'Partition': partition_date,
                                    'Date': datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d"),
                                    'ROW_COUNT_NS': row['ROW_COUNT_NS'],
                                    'ROW_COUNT_PG': row['ROW_COUNT_PG']
                                })
                            except:
                                continue
        if not all_partition_dates:
            st.warning("No partition data available.")
            return

        st.session_state.partition_raw = {
            'data': all_data,
            'dates': sorted(all_partition_dates),
            'raw_counts': all_data_with_counts
        }

    if st.session_state.get('partition_raw'):
        st.subheader("Partition Date Filtering")
        min_date = min(st.session_state.partition_raw['dates'])
        max_date = max(st.session_state.partition_raw['dates'])

        col1, col2 = st.columns(2)
        with col1:
            start_partition = st.date_input("Start partition date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_partition = st.date_input("End partition date", max_date, min_value=min_date, max_value=max_date)

        selected_dates = st.multiselect(
            "Or select specific dates",
            options=st.session_state.partition_raw['dates'],
            format_func=lambda d: d.strftime("%Y-%m-%d")
        )

        filtered_data = []
        filtered_raw_counts = []
        use_date_range = len(selected_dates) == 0

        for item in st.session_state.partition_raw['data']:
            partition_date = item['PartitionDate']
            if (use_date_range and start_partition <= partition_date <= end_partition) or (not use_date_range and partition_date in selected_dates):
                filtered_data.append(item)

        for item in st.session_state.partition_raw['raw_counts']:
            partition_date = item['Partition']
            if (use_date_range and start_partition <= partition_date <= end_partition) or (not use_date_range and partition_date in selected_dates):
                filtered_raw_counts.append(item)

        if not filtered_data:
            st.warning("No data matches the selected partition date filters.")
            return

        df = pd.DataFrame(filtered_data)
        pivot_df = df.pivot_table(index=['Pipeline Name', 'Table Name', 'Partition'], columns='Date', values='Difference', aggfunc='first').reset_index()
        formatted_dates = [datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates]

        for date in formatted_dates:
            if date not in pivot_df.columns:
                pivot_df[date] = float('nan')

        pivot_df = pivot_df[['Pipeline Name', 'Table Name', 'Partition'] + formatted_dates]
        for date_col in formatted_dates:
            pivot_df[date_col] = pivot_df[date_col].apply(format_difference)
        st.session_state.partition_data = pivot_df

        # Abnormal table
        df_counts = pd.DataFrame(filtered_raw_counts)
        abnormal_pivot = df_counts.pivot_table(index=['Pipeline Name', 'Table Name', 'Partition'], columns='Date', values=['ROW_COUNT_NS', 'ROW_COUNT_PG'], aggfunc='first')
        abnormal_df = pd.DataFrame()

        for date in formatted_dates:
            if ('ROW_COUNT_NS', date) in abnormal_pivot.columns and ('ROW_COUNT_PG', date) in abnormal_pivot.columns:
                temp_df = abnormal_pivot[[('ROW_COUNT_NS', date), ('ROW_COUNT_PG', date)]].copy()
                temp_df.columns = ['ROW_COUNT_NS', 'ROW_COUNT_PG']
                temp_df['Abnormal'] = temp_df.apply(calculate_abnormal_behavior_partition, axis=1)
                abnormal_df[date] = temp_df['Abnormal']

        abnormal_df = abnormal_df.reset_index()
        st.session_state.abnormal_df = abnormal_df

    if st.session_state.get('partition_data') is not None:
        st.dataframe(st.session_state.partition_data, use_container_width=True, height=600)

        st.subheader("Abnormal Behavior Detection")

        # Identify max percentage from abnormal_df
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
                            if col not in ['Pipeline Name', 'Table Name', 'Partition']]
            
            percentage_df = st.session_state.abnormal_df[date_columns].map(extract_percentage)
            max_percent = percentage_df.max().max()
            max_percent = round(max_percent, 1) if max_percent else 10.0

            # Slider with max from data
            threshold = st.slider(
                "Move slider to filter by minimum percentage difference (%)",
                min_value=0.0,
                max_value=max_percent,
                value=0.0,
                step=0.1,
                key="partition_threshold"
            )

            mask = percentage_df.ge(threshold).any(axis=1)
            filtered_abnormal = st.session_state.abnormal_df[mask]

            st.dataframe(
                filtered_abnormal,
                use_container_width=True,
                height=600
            )
