# Data Visualizer & Analyzer v1.0

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-1.0-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A powerful data visualization and analysis tool that helps users explore, analyze, and discover trends in their data through intuitive visualizations and comprehensive analysis methods.

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Format Support** | Read data from various file formats |
| **Advanced Filtering** | Column and table-level filtering capabilities |
| **Comparative Analysis** | Inter and intra-table comparison methods |
| **Data Validation** | Verify correct datatype assignments |
| **Visualization Suite** | Multiple chart types for different analytical needs |
| **Trend Discovery** | Identify potential overlooked patterns in data |

## Supported File Formats

| Extension | Format | Status |
|-----------|--------|--------|
| `.csv`    | Comma-Separated Values | ✅ Fully Supported |
| `.xlsx`   | Excel Workbook | ✅ Fully Supported |
| `.xls`    | Excel 97-2003 | ✅ Fully Supported |
| `.json`   | JavaScript Object Notation | ✅ Fully Supported |
| `.ods`    | OpenDocument Spreadsheet | ✅ Fully Supported |

## Implemented Visualizations

- ✅ Box Plots
- ✅ Scatter Graphs
- ✅ Line Charts
- ✅ Bar Charts
- ✅ Histograms

## Upcoming Features

- 🚧 Pie Charts (In Development)
- 🚧 Improved Data Organization
- 🚧 Advanced Relational Methods
- 🚧 User Interface Implementation
- 🚧 Code Refactoring for User-Friendliness

## Project Purpose

This tool was developed to:
- Make data study and analytics more accessible through automation
- Provide a fun, interactive way to explore datasets
- Reduce manual effort in preliminary data analysis
- Help users discover insights they might otherwise overlook

## Usage

```python
# Sample usage code will go here
from data_analyzer import DataVisualizer

analyzer = DataVisualizer("data.csv")
analyzer.show_visualizations()



