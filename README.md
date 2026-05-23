# Multi-Source Remote Sensing Fusion & Daily Vegetation Index Time Series Reconstruction
Code repository for reproducing analyses and figures in the manuscript:
**Integrating multi-source remote sensing phenological dynamic features and interpretable XGBoost-SHAP for winter wheat yield estimation across the North China Plain**

## Overview
This repository contains the custom code used for **multi-source satellite data fusion**, linear regression-based cross-calibration, Ensemble Kalman Filter (EnKF) time-series smoothing, and full-band vegetation index calculation for long-term daily remote sensing analysis. The workflow integrates Landsat 8, MODIS, and Sentinel-2 surface reflectance to generate gap-free, smoothed vegetation index time series for agricultural monitoring.

## Repository Structure
├── GEE.py                                      # GEE script: Multi-source satellite data extraction (L8/MODIS/S2)

├── Calculation of remote sensing index feature values.py  # Python script: Fusion, calibration, EnKF smoothing, and index calculation

├── Points_2014201510.csv                       # Raw GEE output:  surface reflectance time series

├── 多源融合校正.csv                             # Intermediate: Multi-source fused & MODIS-calibrated daily reflectance

├── EnKF时序优化.csv                             # Intermediate: EnKF-smoothed daily reflectance

├── 全作物指数.csv                               # Final: Full set of quality-controlled vegetation indices

└── README.md # Repository documentation


## Reproducibility Instructions
- Raw Data Extraction
Run GEE.py in the Google Earth Engine Code Editor.
Output: 23Points_2014201510.csv (multi-source surface reflectance at 23 study points).
- Multi-Source Fusion & Calibration
Run the first part of Calculation of remote sensing index feature values.py.
Input: 23Points_2014201510.csv
Output: 多源融合校正.csv (MODIS-calibrated, gap-filled daily reflectance).
- EnKF Time Series Smoothing
Run the EnKF section of Calculation of remote sensing index feature values.py.
Input: 多源融合校正.csv
Output: EnKF时序优化.csv (smoothed daily reflectance).
- Vegetation Index Calculation & Quality Control
Run the index computation and correction sections of the Python script.
Input: EnKF时序优化.csv
Output: 全作物指数.csv (final, range-constrained vegetation index time series).

## Core Functions
- Multi-source satellite data extraction (Landsat 8, MODIS, Sentinel-2)
- Linear regression cross-calibration to harmonize MODIS data with high-resolution sensors
- Daily gap-filling via linear interpolation
- Ensemble Kalman Filter (EnKF) for temporal smoothing and noise reduction
- Full vegetation index computation: NDVI, EVI2, SAVI, OSAVI, GNDVI, CIgreen, MSI, NDWI, RVI, PRI, and custom LMI
- Physically based index range correction and post-smoothing
- Batch export of intermediate and final products as CSV files







你还可以做哪些其他类型的数据分析？
你是如何进行数据处理的？
你可以处理哪些类型的数据？
