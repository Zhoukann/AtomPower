import os
import re
import sys
import csv
import pandas as pd
from sympy import Symbol

design_name = sys.argv[1]
testbench = sys.argv[2]

# input
reg_features_file  = f"./Design/Structure/{design_name}_reg_features.csv"
raw_reg_waveform_file = f"/export/home/stu5/RISCV/{design_name}Config/sim_rtl/csv/{design_name}_{testbench}.fsdb.csv"

# temp
temp_file1 = f"./{testbench}/{design_name}_{testbench}_reg_waveform_temp1.csv"
temp_file2 = f"./{testbench}/{design_name}_{testbench}_reg_waveform_temp2.csv"
temp_file3 = f"./{testbench}/{design_name}_{testbench}_reg_waveform_temp3.csv"

# output
reg_waveform_file = f"./{testbench}/{design_name}_{testbench}_reg_waveform.csv"


fs = Symbol("fs")
ps = Symbol("ps")
ns = Symbol("ns")
us = Symbol("us")
ms = Symbol("ms")
s = Symbol("s")

conversion_factors = {
    'fs': 1e-6,
    'ps': 1e-3,
    'ns': 1,
    'us': 1e3,
    'ms': 1e6,
    's' : 1e9
}

def convert_to_ns(value ,unit):
    if unit not in conversion_factors:
        raise
        ValueError(f"Unsupport unit:{unit}")
    return int(value * conversion_factors[unit])


def re_write(line,equal,k):
    line = line.strip().split(',')

    if k == 1:

        reg_name = []
        for n in range(len(line)):

            if equal[n] == 1:
                continue

            part = line[n]
            if len(part.split('[')) == 1:
                reg_name.append(part)

            if len(part.split('[')) == 2:
                reg_width = int(part.split('[')[1].split(']')[0].split(':')[0]) + 1
                for i in range(reg_width):
                    reg_name.append(f"{part.split('[')[0]}[{reg_width - 1 - i}]")

            if len(part.split('[')) == 3:
                dim1_range = part.split('][')[0].split('[')[-1]
                dim2_range = part.split('][')[1].split('[')[-1].split(']')[0]

                dim1_start,  dim1_end = map(int, dim1_range.split(':')[1::-1])
                dim2_start,  dim2_end = map(int, dim2_range.split(':')[1::-1])

                dim1_size = dim1_end - dim1_start + 1
                dim2_size = dim2_end - dim2_start + 1

                for i in range(dim1_size):
                    for j in range(dim2_size):
                        reg_name_formatted = f"{part.split('[')[0]}[{dim1_end-i}][{dim2_end-j}]"
                        reg_name.append(reg_name_formatted)
        return reg_name

    if k == 0:
        data = []
        data.append(line[0])
        for n in range(len(line[1:])):

            if equal[n+1] == 1:
                continue

            part = line[n+1]
            data_temp = list(part)
            for j in range(len(data_temp)):
                    data.append(data_temp[j])
        return data


def reg_waveform_process1():

    #unit_number = 0
    #unit = ""
    line_num = 0
    unit_number = 1.0
    unit = "ps"

    print("############## Processing start ##############")

    keywords = []
    with open(reg_features_file,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            keywords.append(row[0].split("[")[0])

    columns = pd.read_csv(raw_reg_waveform_file, nrows = 0).columns.tolist()

    #print(columns[0])
    #match = re.search(r"Time\(([\d.]+)([a-zA-Z]+)\)", columns[0].strip())

    #if match:
    #    unit_number = float(match.group(1))
    #    unit = match.group(2)


    #print(f"Time unit is {columns[0]}")

    matched_columns = [columns[0]]

    for col in columns[1:]:
        #part = col.rsplit('/',1)[-1].split('[',1)[0]
        #if any(keyword in col for keyword in keywords):
        if col.split("[")[0] in keywords:
            #print(f"{part}: register")
            matched_columns.extend([col])
        #else:
            #print(f"{part}: not a register")

    if len(matched_columns) < 1:
        print("No register in this file! Please check your previous work.")
        return None

    equal = [1]*len(matched_columns)
    equal_con = pd.read_csv(raw_reg_waveform_file, nrows = 1, dtype = str)[matched_columns].values.flatten().tolist()


    print("1st step processing, select registers")
    with open(temp_file1,'w') as f:
        
        f.write("Time," + ','.join(matched_columns[1:]) + '\n')

        for chunk in pd.read_csv(raw_reg_waveform_file,chunksize = 500, dtype = str):

            line_num = line_num + 500
            #print(f"Processing data of {int(chunk.iloc[0,0])/1_000}ns to {int(chunk.iloc[-1,0])/1_000}ns")
            print(f"Processing data of {convert_to_ns(int(chunk.iloc[0,0])*unit_number,unit)}ns to {convert_to_ns(int(chunk.iloc[-1,0])*unit_number,unit)}ns")
            # the unit!

            filtered_chunk = chunk[matched_columns].copy()
            filtered_chunk[columns[0]] = chunk[columns[0]].apply(lambda x: convert_to_ns(int(x)*unit_number, unit))
            filtered_chunk.to_csv(f, header = False, index = False, mode = 'a')

            for i, col in enumerate(matched_columns):
                if equal[i] == 0:
                    continue
                elif len(filtered_chunk[col].unique()) == 1 and filtered_chunk[col].unique()[0] == equal_con[i]:
                    continue
                else:
                    equal[i] = 0

    with open(temp_file1,'r') as f:
        line = f.readline()
        print(f"After 1st step processing, {len(line.strip().split(','))} registers remain.")
    
    print("2nd step processing, delete no-switching registers, and one column for one bit")
    with open(temp_file2,'w') as f2, open(temp_file1,'r') as f1:

        #print("Registers rewriting:")
        line = f1.readline()
        reg_name = re_write(line, equal, 1)
        df = pd.DataFrame([reg_name])
        df.to_csv(f2,index = False,header = False)

        n = 0
        for chunk in pd.read_csv(temp_file1, chunksize = 500, skiprows = 1, dtype = str, header = None):
            n = n + 500
            sys.stdout.write(f"\r{n}/{line_num}")
            first_column = chunk.iloc[:,0]
            filtered_chunk = chunk.iloc[:,1:].loc[:, [i for i in range(1,len(equal)) if equal[i] == 0]]
            new_data = filtered_chunk.apply(lambda x: ''.join(x.astype(str)), axis=1).apply(list)
            new_data_df = pd.DataFrame(new_data.tolist())
            new_df = pd.concat([first_column.reset_index(drop = True), new_data_df], axis = 1)
            #final_df = new_df.iloc[1::2]###############################
            new_df.to_csv(f2,index = False,header = False)
            #final_df.to_csv(f2,index = False,header = False)

    #os.remove(temp_file1)

    with open(temp_file2,'r') as f:
        line = f.readline()
        print(f"\nAfter 2nd processing, {len(line.strip().split(','))} separate registers remain.")
    #print(f"############## Processing finished ##############")


def reg_waveform_process2():

    last_row = None
    header_written = False

    line_num = 0

    print("3rd step processing, switch signal --> 1, no-switch signal --> 0")
    with open(temp_file2,'r') as f1, open(temp_file3,'w') as f2:
        for chunk in pd.read_csv(temp_file2, chunksize = 500):

            line_num = line_num + 500
            print(line_num)

            time_col = chunk.iloc[:,0]

            if last_row is not None:
                chunk = pd.concat([last_row,chunk], ignore_index = True)
        
            result_chunk = pd.DataFrame()
            result_chunk[chunk.columns[0]] = chunk.iloc[:,0]

            for column in chunk.columns[1:]:
                values = chunk[column]
                changes = values.ne(values.shift(1)).astype(int)
                result_chunk[column] = changes

            if last_row is None:
                result_chunk.iloc[0,1:] = 0
            else:
                result_chunk = result_chunk.iloc[1:]

            last_row = chunk.iloc[[-1]]

            result_chunk.to_csv(f2, mode = 'a', header = not header_written, index = False)

            header_written = True

    print(f"############## Processing finished ##############")

# set the time at the middle of clock edge
def reg_waveform_process3():

    last_row = None
    header_written = False

    line_num = 0

    with open(temp_file2,'r') as f1, open(temp_file3,'w') as f2:
        for chunk in pd.read_csv(temp_file2, chunksize = 500):

            original_columns = chunk.columns
            first_col_name = original_columns[0]
            if header_written is not True:
                print(f"\nUnit is: {first_col_name}")

            line_num = line_num + 500
            print(line_num)

            data_cols = chunk.iloc[:,1:]
            time_col = chunk.iloc[:,0]

            new_time_data = []
            new_data_rows = []

            for i in range(len(time_col)-1):
                new_time = int((time_col.iloc[i] + time_col.iloc[i+1])/2)
                new_time_data.append(new_time)

                if last_row is not None:
                    new_row = (data_cols.iloc[i] != last_row).astype(int)

                else:
                    new_row = pd.Series([0] * data_cols.shape[1], index = data_cols.columns)
                
                new_data_rows.append(new_row)

                last_row = data_cols.iloc[i]

            last_row = data_cols.iloc[-1]

            new_chunk = pd.DataFrame(new_data_rows, columns=data_cols.columns)
            new_chunk.insert(0, first_col_name, new_time_data)

            new_chunk.to_csv(f2, header=not header_written, index=False)
            header_written = True


def check_process():

    df = pd.read_csv(temp_file3)
    dfs = pd.read_csv(reg_features_file)

    available_regs = dfs["reg_name"].tolist()
    check = df.columns[1:]
    available_columns = [col for col in available_regs if col in check]

    df = df[[df.columns[0]] + available_columns]


    df = df[df.iloc[:,1:].sum(axis=1) > 1]


    df = df.loc[:,df.nunique() > 1]


    subset_columns = df.columns[1:-1]
    df = df.drop_duplicates(subset = subset_columns)

    df.to_csv(reg_waveform_file, index = False)

    with open(reg_waveform_file,'r') as f:
        line = f.readline()
        print(f"{len(line.strip().split(','))-2} separate registers remain.")



reg_waveform_process1()
reg_waveform_process2()
check_process()

"""
Waveform in input file:
    A/B/a_reg[2:0],A/b_reg[1:0],C/c_signal
    111,11,1
    100,11,0
    100,11,1

Waveform in file "temp_file1":
    A/B/a_reg[2:0],A/b_reg[1:0]
    111,11
    100,11
    100,11
select registers in file "reg_features_file" 

Waveform in file "temp_file2":
    A/B/a_reg[2],A/B/a_reg[1],A/B/a_reg[0]
    1,1,1
    1,0,0
    1,0,0
delete no-switching registers
one column for one bit

Waveform in file "temp_file3":
    A/B/a_reg[2],A/B/a_reg[1],A/B/a_reg[0]
    0,0,0
    0,1,1
    0,0,0
switch: 1, no switch: 0

Waveform in file "reg_waveform_file":
    A/B/a_reg[2],A/B/a_reg[1],A/B/a_reg[0]
    0,1,1
delete no-switching column and duplicate row
"""