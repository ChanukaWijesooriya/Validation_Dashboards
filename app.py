import streamlit as st
import volume_validation, partition_validation

st.set_page_config(page_title="Validation Dashboard", layout="wide")
st.title("Validation Dashboard")

# Initialize session state
if 'selected_pipeline' not in st.session_state:
    st.session_state.selected_pipeline = "All"
if 'selected_table' not in st.session_state:
    st.session_state.selected_table = "All"
if 'volume_data' not in st.session_state:
    st.session_state.volume_data = None
if 'partition_data' not in st.session_state:
    st.session_state.partition_data = None
if 'partition_raw' not in st.session_state:
    st.session_state.partition_raw = None

# Create tabs
tab1, tab2 = st.tabs(["Volume Validation", "Partition Validation"])

with tab1:
    # Clear partition-related session state when in volume tab
    if 'partition_data' in st.session_state:
        del st.session_state.partition_data
    if 'partition_raw' in st.session_state:
        del st.session_state.partition_raw
    volume_validation.create_volume_table()

with tab2:
    # Clear volume-related session state when in partition tab
    if 'volume_data' in st.session_state:
        del st.session_state.volume_data
    partition_validation.create_partition_table() 
