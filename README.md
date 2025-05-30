# Data Validation Dashboard

## Overview

This Streamlit application serves as an internal validation tool for engineering teams to ensure data consistency between PostgreSQL (source) and Snowflake (target) pipelines. The dashboard provides dynamic validation capabilities across two important sectors:

1. Volume Validation: Compares total row counts between source and target systems

2. Partition Validation: Verifies row counts at the partition level for incremental loads

## Key Features:

- Interactive Dashboard: User-friendly interface with filtering capabilities

- Historical Trend Analysis: View data discrepancies across 5 most recent validation runs

- Anomaly Detection: Automatic flagging of significant data discrepancies

- Pipeline-Specific Insights: Drill down into specific pipelines and tables

- Partition-Level Validation: Examine data consistency by date partitions

## Project Purpose

This tool was specifically created for:

- Internal engineering decision-making

- Ensuring data consistency across PostgreSQL (source) and Snowflake (target) pipelines

- Validating both historical/full-load migrations (volume validation)

- Verifying incremental loads (partition validation)

- Identifying and troubleshooting data discrepancies early in the migration process

## Validation Methods

### Volume Validation

- Compares total row counts (ROW_COUNT_NS vs ROW_COUNT_PG) for entire tables

- Ideal for validating full-load data migrations

- Displays historical trends of row count differences

- Highlights tables with significant discrepancies

### Displayed Data:

- Pipeline name and table name

- Date of validation run

- Row count difference (Snowflake vs PostgreSQL)

- Visual indicators for data anomalies

### Partition Validation

- Compares row counts at the partition level (PARTITION_COLUMN_VALUE)

- Essential for validating incremental data loads

- Allows filtering by date ranges or specific partitions

- Identifies gaps or mismatches in time-bound data slices

### Displayed Data:

- Pipeline name, table name, and partition value

- Date of validation run

- Partition-level row count differences

- Visual indicators for partition anomalies

## Anomaly Detection

- The application automatically flags potential issues using a tiered alert system:

🔴 Red: > 2% difference (Critical discrepancy)

🟡 Yellow: 1-2% difference (Warning)

🟢 Green: < 1% difference (Normal variation)

0: No difference or insufficient data

## Usage Instructions:

1. Select Pipeline Group: Choose "All" or a specific pipeline

2. Select Table: Filter to "All" tables or a specific table

3. For Partition Validation:

4. Set date range or select specific partitions

5. Click "Search" to refresh results with current filters

## Technical Implementation

- Frontend: Streamlit web application

- Data Structure: CSV files organized by date in data/ directory

- Validation Logic:

    - Volume validation: volume_validation.py

    - Partition validation: partition_validation.py

- Session Management: Maintains user selections across interactions

## Data Directory Structure

The application expects data in the following structure:

![Image](https://github.com/user-attachments/assets/6dacb6de-e4a9-4567-842f-64f1b00c638a)

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Pathlib


## File Structure

![Image](https://github.com/user-attachments/assets/6365736c-c0f0-4070-8256-3a45caa350f5)

# Running the Application

`streamlit run app.py`
