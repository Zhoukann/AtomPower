# Power Modeling and Prediction Framework for Rocket / BOOM Cores

This repository contains scripts and preliminary workflows for **parsing elaborated SystemVerilog designs**, **processing FSDB waveform data**, and **training cross-core power prediction models** (e.g., SmallBOOM → MediumBOOM).

Due to GitHub file size limitations, only part of the full pipeline is currently runnable. Nevertheless, **all core scripts and implementation logic are already included**, and the repository will be incrementally completed with reduced or processed datasets.

---

## 1. Design Parsing (Runnable)

The first step parses an **elaborated SystemVerilog design** and extracts **structural features** (e.g., module hierarchy, registers, connections) for later modeling.

### Usage
```bash
cd Rocket/Design
python3 1_Parse.py Rocket.elab.v
