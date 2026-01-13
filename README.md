# Power Modeling and Prediction Framework

This repository contains scripts and preliminary workflows for **parsing elaborated SystemVerilog designs**, **processing FSDB waveform data**, and **training cross-core power prediction models** (e.g., SmallBOOM → MediumBOOM).

Due to GitHub file size limitations, only part of the full pipeline is currently runnable. Nevertheless, **all core scripts and implementation logic are already included**, and the repository will be incrementally completed with reduced or processed datasets.


## 1. Design Parsing (Runnable)

The first step parses an **elaborated SystemVerilog design** and extracts **structural features** (e.g., module hierarchy, registers, connections) for later modeling.

### Usage
```bash
cd Rocket/Design
python3 1_Parse.py Rocket.elab.v

## 2. Waveform Processing Scripts (Raw / Data-Dependent)

The following scripts are provided as **raw processing scripts** for handling original **FSDB waveform files** and generating **intermediate features** and **power-related weights** for training:

- `1_reg_waveform.py`
- `2_power_waveform.py`
- `3_module_waveform.py`
- `4_pre_regression.py`
- `5_waveform_same.py`
- `6_regression.py`

### Intended Workflow

These scripts collectively perform the following steps:

1. Parse FSDB waveform files  
2. Extract register-level activity  
3. Aggregate module-level activity  
4. Align switching activity with power traces  
5. Generate regression-ready datasets  

## 3. Cross-Core Power Prediction: B2Bpred

`B2Bpred` implements a **Boom-to-Boom power prediction task**, where a power model is trained on one core configuration and then applied to another.

### Current Setup

- **Training core**: SmallBOOM  
- **Target core**: MediumBOOM  

This task evaluates the **transferability of power models across microarchitectural scales**, assuming partially shared design patterns but different performance and power characteristics.

