# NAFLD Biopsy Analysis Pipeline

A comprehensive digital pathology pipeline for automated analysis of Non-Alcoholic Fatty Liver Disease (NAFLD) whole slide images. This pipeline performs quality control, stain normalization, tissue segmentation, feature extraction, and comprehensive analysis to support clinical research and biomarker discovery.

## Project Structure
nafld_biopsy_analysis
├── analysis
│   ├── __init__.py
│   ├── biomarker_analysis.py
│   ├── clinical_correlation.py
│   └── subtype_discovery.py
├── annotation
│   ├── __init__.py
│   ├── tools.py
│   └── validation.py
├── config
│   ├── __init__.py
│   ├── parameters.py
│   └── paths.py
├── config.yaml
├── data
│   ├── __init__.py
│   ├── loaders.py
│   ├── preprocessing.py
│   └── quality_control.py
├── features
│   ├── __init__.py
│   ├── extraction.py
│   ├── selection.py
│   └── spatial.py
├── models
│   ├── __init__.py
│   ├── classification.py
│   ├── detection.py
│   ├── segmentation.py
│   └── training.py
├── requirements.txt
├── run_pipeline.py
├── structure.txt
├── utils
│   ├── __init__.py
│   ├── helpers.py
│   └── loggers.py
└── visualization
    ├── __init__.py
    ├── clinical_plots.py
    ├── eature_plots.py
    ├── model_plots.py
    └── qc_plots.py


## Create the following directory structure:
/home/ubuntu/nafld_data/
├── raw/                   # Original WSI files (.svs, .ndpi, .tif, .tiff)
├── clinical/              # Clinical data (clinical_data.csv)
├── controls/              # Reference slides for stain normalization
└── processed/             # Auto-created by pipeline

## Edit config.yaml with your paths:
data:
  base_dir: "/home/ubuntu/nafld_data"
  clinical_data: "/home/ubuntu/clinical/clinical_data.csv"


## Run complete pipeline
python run_pipeline.py --config config.yaml --steps all

## Run specific steps
python run_pipeline.py --config config.yaml --steps qc normalization segmentation