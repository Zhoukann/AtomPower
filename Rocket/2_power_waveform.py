# Calculate per-cycle power dissipation
import re
import os
import sys
import pandas as pd
from sympy import Symbol

design_name = sys.argv[1]
testbench = sys.argv[2]

# input
reg_waveform_file = f"./{testbench}/{design_name}_{testbench}_reg_waveform.csv"
reg_features_file  = f"./Design/Structure/{design_name}_reg_features.csv"
stable_power_file = f"./Design/stable_power.csv"
raw_power_waveform_file = f"/export/home/stu5/RISCV/{design_name}Config/pt/{testbench}/primetime_px.fsdb.csv" 

# temp
power_waveform_temp_file = f"./{testbench}/{design_name}_{testbench}_power_waveform_temp.csv"

# output
power_waveform_file = f"./{testbench}/{design_name}_{testbench}_power_waveform.csv"



def power_waveform_process1():

    strip_path = "/TestDriver/testHarness/chiptop0/system/tile_prci_domain/element_reset_domain_boom_tile"

    def column_name(col):
        if col == "Time(1n)":
            return "Time"
        if len(col.split("/")) > 1:
            return strip_path + col.split("Pc(")[0] + col.split("Pc(")[1].split(")")[0] + "/"
        else:
            return strip_path + col.split("Pc(")[0].split(")")[0] + "/"

    def convert_to_nW(x):

        power_with_unit = x

        if  power_with_unit[-2].isdigit() == True:
            power_value = float(power_with_unit[:-1])
            power_unit = power_with_unit[-1:]# just 'W'
        else:
            power_value = float(power_with_unit[:-2])
            power_unit = power_with_unit[-2:]

        if power_unit == 'nW':
            power_value = power_value 
        elif power_unit == 'uW':
            power_value = power_value * 1e3
        elif power_unit == 'mW':
            power_value = power_value * 1e6
        elif power_unit == 'W':
            power_value = power_value * 1e9
        elif power_unit == 'pW':
            power_value = power_value * 1e-3 
        elif power_unit == 'fW':
            power_value = power_value * 1e-6
        else:
            print(f"Unknown power unit: {power_unit}.")
            return None
        return power_value
    

    df = pd.read_csv(raw_power_waveform_file)
    df.columns = [column_name(col) for col in df.columns]

    reg_df = pd.read_csv(reg_features_file)
    registers = reg_df["reg_name"].tolist()

    valid_columns = []

    for col in df.columns:
        if any(col in register for register in registers):
            valid_columns.append(col)

    df = df[["Time"] + valid_columns]

    for col in valid_columns:
        df[col] = df[col].apply(lambda x: convert_to_nW(str(x)))

    df.to_csv(power_waveform_temp_file, index = False)


def power_waveform_process2():

    df = pd.read_csv(power_waveform_temp_file)

    reg_flip_df = pd.read_csv(reg_waveform_file)
    reg_flip_time = reg_flip_df["Time"].tolist()

    cs_power_df = pd.read_csv(stable_power_file)
    cs_power_dict = dict(zip(cs_power_df["0"],cs_power_df["1"])) 

    time_col = "Time"

    new_df = pd.DataFrame()

    new_df[time_col] = reg_flip_time

    for col in df.columns:
        if col != time_col:

            power_column = df[col]
            new_power_column = []

            for flip_time in reg_flip_time:
                
                indices = df[df[time_col] == flip_time].index
                i = indices[0]

                power_sum = power_column[i+1] + power_column[i+2] - cs_power_dict.get(col,0)

                new_power_column.append(power_sum)

            new_df[col] = new_power_column

    new_df.to_csv(power_waveform_file, index = False)
    


power_waveform_process1()  
power_waveform_process2()                  
