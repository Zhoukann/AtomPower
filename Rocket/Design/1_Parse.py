# File for parsing .elab.v files made by synopsys design compiler
import re 
import sys
import csv
import copy
import json
import numpy as np
import pandas as pd
from functools import reduce
from collections import OrderedDict
import time
###########################################################

# the top module name and design name and dont parse modules
top_module_name = 'Rocket'
design_name = 'Rocket'

module_dont_parse = []
#module_dont_parse = ['RenameFreeList','RenameFreeList_1','FpPipeline','BoomCore']

# file to be parsed
filename = sys.argv[1]

# output
structure_tree_file = f"./Structure/{design_name}_reg_structure_tree.txt"
reg_features_temp_file = f"./Structure/{design_name}_reg_features_temp.csv"
reg_crossing_file = f"./Structure/{design_name}_reg_crossing.json"

###########################################################

# recursion and tree level
recursion = 1
level = 0

# list of module objects
modules = []
# list of modules name
modules_name = []
# the module being parsed, be added to "longname" of gtech objects
# to ensure each unit has one and noly one separate gtech object and structure object 
modulename_parsing = ''

# list of structure trees roots (regs and top module inputs)
top_level_parents = []
top_level_parents_name = []

# list of structures objects
structures = []
# list of structures name
structures_name = []

# 
reg_structures_stack = []
reg_stack = []

feature = dict()

gtech_stack = {
    "gtech_or2" : [],
    "gtech_and2" : [],
    "gtech_xor2" : [],
    "gtech_not" : [],
    "gtech_buf" : [],
    "select_op" : [],
    "mux_op" : [],
    "comp_op" : [],
    "sub_op" : [],
    "add_op" : [],
    "mult_op" : [],
    "div_op" : [],
    "shift_op": []
}


# lists setting module environment
regs            = np.array([])
nots            = np.array([])
bufs            = np.array([])
and2s           = np.array([])
or2s            = np.array([])
muxes           = np.array([])
selects         = np.array([])
connects        = np.array([])
inputs          = np.array([])
outputs         = np.array([])
dependencies    = np.array([])
shifters        = np.array([])
comparators     = np.array([])
xor2s           = np.array([])
multipliers     = np.array([])
subtractors     = np.array([])
b_shifters      = np.array([])
adders          = np.array([])
shift_adders    = np.array([])
divisors        = np.array([])
assigns         = np.array([]) 
unknowns         = np.array([])

# gate counts
reg_n        = 0
not_n        = 0
buf_n        = 0
and2_n       = 0
or2_n        = 0
mux_n        = 0
select_n     = 0
shift_n      = 0
comp_n       = 0
xor2_n       = 0
mult_n       = 0
sub_n        = 0
b_shift_n    = 0
add_n        = 0
shift_add_n  = 0
div_n        = 0
input_n      = 0

# Top 
def run_parse_elab(filename,top_module_name):

    start_time = time.time()

    global modules
    global top_level_parents
    global structures

    parse_file(filename)

    print("markA")

    for m in modules:
        print(f"markB{m.name}")
        m.set_connection_points()

    process_dependencies()     
    print("markC")

    for m in modules:
        if(m.name == top_module_name):
            print("markD")
            connect_structure(m)
            print("markE")
            count_gates(m)
            print("markF")
            print_gates()
            print("markG")

    #return True

    sys.stdout = open(structure_tree_file,'w')
    for s in top_level_parents:
        print(s.represented_object_handle.name)
        print(s)
        #print(s.represented_object_handle)
    sys.stdout = sys.__stdout__

    # Delete the extra '_reg' in .elab.v file
    # Examples of registers' names in .elab.v file:        
    # x      ===> x_reg
    # x_reg  ===> x_reg_reg
    # x[1:0] ===> x_reg[1]  x_reg[0]
    # In this part we delete the last '_reg' and the index.
    #sys.stdout = open('./data/reg_list.txt','w')
    #for s in top_level_parents:
    #    string = s.represented_object_handle.longname.split('[')

        #match = re.match(r'(\w+)_reg(?:\[\d+\])?.*',string)
        #if match:
        #    print(match.group(1))
        #else:
        #    print(string)

    #    print(string[0])

    #sys.stdout = sys.__stdout__

    header = ['reg_name']
    first_row = feature_in_one(top_level_parents[0])

    for group_name, feature_dict in first_row.items():
        for feature_name in feature_dict.keys():
            header.append(f"{group_name}_{feature_name}")
    
    with open(reg_features_temp_file, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for module_name in modules_name:
        if module_name not in module_dont_parse:
        #if module_name == module_to_parse:
            features(module_name)
    
    #print(f"{len(top_level_parents)} input/reg objects (there may be something wrong with the input, I am fixing it but it too hard)")
    print(f"{len(structures)} input/output/gtech/unknown objects")


    data = {}
    added_cross_terms = set()

    for s in top_level_parents:
        print(s.represented_object_handle.longname)
        cross_term = frozenset([parent.represented_object_handle.longname for parent in s.reg_parents])
        if cross_term not in added_cross_terms:
            added_cross_terms.add(cross_term)
            data[s.represented_object_handle.longname] = list(cross_term)
    
    with open(reg_crossing_file,"w") as f:
        json.dump(data,f,indent=4)


    end_time = time.time()

    run_time = end_time - start_time
    print(run_time)

    return modules, top_level_parents



# Start parsing .elab.v file
def parse_file(path):

    global modules
    global modules_name
    global modulename_parsing

    print ("Start parsing file:" + path)
    object_handle   = 'false'
    objectstring    = "" # the parsing line
    bitwidth = 0

    moduleline = ''
    in_module = False # judge whether is parsing a module
    

    # read file line by line
    with open(path, 'r') as svfile:

        line = svfile.readline()
        linenum = 1

        while line:

            print("linenum: " + str(linenum))
            if line.strip() == '':

                line = svfile.readline()
                linenum = linenum + 1
                continue

            if (in_module):

                if (find_endmodule(line)):

                    in_module = False
                    module_handle = module(modulename_parsing) # create a new "module" object
                    module_handle.set_lists() 
                    empty_global_lists() 
                    module_handle.connection_point_string = moduleline
                    connect_assigns()     
                    modules.append(module_handle)
                    modules_name.append(modulename_parsing)

                else:

                    offset = 0
                    key, match = parse_line(line, 'false')
                    #print("key: " + key)
                    #print(match)
                    if(match == None):
                        line = svfile.readline()
                        linenum = linenum + 1
                        continue
                    if match.group(1) == 'input' or match.group(1) == 'output' or match.group(1) == 'wire':
                        if match.group(2) != None:
                            bitwidth = int(match.group(2)) - int(match.group(3)) + 1
                            line = " ".join(line.split()[2:])
                            offset = int(match.group(3))
                        else:
                            bitwidth = None 
                            line = " ".join(line.split()[1:])
                    else:
                        if(key == 'dep'): 
                            if match.group(1) in modules_name:
                                object_handle = make_object(match.group(2), key)
                                object_handle.modulename = match.group(1)
                                print("Make dependency: "+match.group(2)+" of module "+match.group(1))
                            else:
                                print("Unknown object/module: " + match.group(1))
                                object_handle = make_object(match.group(2), 'unknown_object')
                        else:
                            object_handle = make_object(match.group(1), key)
                        
                    # get new lines until all of object(s) is in one string
                    in_object = True
                    while in_object:
                        for k, rx in rx_dict_end.items():
                            objectstring = objectstring + line
                            match = rx.search(line)
                            if match:
                                # found end, leave while loop
                                in_object = False
                            else:
                                # read new line until end is found
                                line = svfile.readline()
                                linenum = linenum + 1
                
                    #print("objectstring: " + objectstring)
                    objectstring = "".join(objectstring.split())# remove charaters like space, tab, enter

                    if (key == 'input' or key == 'output' or key == 'wire'):
                        objectstring = objectstring.strip(";")
                        objectstring = objectstring.translate({ord(i): None for i in '}{ '})
                        objectlist = objectstring.split(',')
                        for name in objectlist:
                            object_handle = make_object(name, key)
                            if (bitwidth != None): 
                                object_handle.width = bitwidth
                                object_handle.widthoffset = offset
                                object_handle.init_connection_nodes()
                            #print ("made new connection object with name "+name + " and width "+ str(bitwidth))
                    else:               
                        parse_line(objectstring, object_handle)
                        #print("mark")
                    # when done with an object, empty object string
                    objectstring = ""

            else:
                modulename_parsing, in_module = find_module(line)
                if (modulename_parsing == None):
                    line = svfile.readline()
                    linenum = linenum + 1
                    continue
                print("Start parsing module: " + modulename_parsing)

                # find the end of module declaration
                module_declaration_ongoing = True
                while module_declaration_ongoing:
                    for k, rx in rx_dict_end.items():
                        objectstring = objectstring+line
                        match = rx.search(line)
                        if match:
                            module_declaration_ongoing = False
                            moduleline = objectstring
                            #print(moduleline)
                        else:
                            line = svfile.readline()
                            linenum = linenum +1
                objectstring = ''
                
            line = svfile.readline()
            linenum = linenum + 1
            #print(linenum)

# start parsing one line
# looking for start of object and internal parameters of object
def parse_line(line, object_handle):

    #print(line)
    #print ("parsing line: \n"+ line + "\n in object:\n" + str(object_handle))
    key = ""
    match = ""

    # look for start of object to determine object type
    if (object_handle == 'false'):
        for key, rx in rx_dict_start.items(): 
            match = rx.search(line)
            if match:
                return key, match  
 
    elif (object_handle.id == 'reg'):
        for key, rx in rx_dict_reg.items():
            match = rx.search(line)
            #print(match.group(2))
            #print("line: " + line)
            #print ("found match for: " + key + " group captured: " +  match.group(1) )
            if (key == 'clear' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.clear = match.group(1)
            elif (key == 'preset' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.preset = match.group(1)
            elif (key == 'next_state' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.next_state = match.group(1)
            elif (key == 'clocked_on' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.clocked_on = match.group(1)
            elif (key == 'data_in' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.data_in = match.group(1)
            elif (key == 'enable' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.enable = match.group(1)
            elif (key == 'Q' and match):
                process_match([match.group(1)], object_handle, 'output', key)
                object_handle.Q = match.group(1)
            elif (key == 'QN' and match):
                process_match([match.group(1)], object_handle, 'output', key)
                object_handle.QN = match.group(1)
            elif (key == 'synch_clear' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.synch_clear = match.group(1)
            elif (key == 'synch_preset' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.synch_preset = match.group(1)
            elif (key == 'synch_toggle' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.synch_toggle = match.group(1)
            elif (key == 'synch_enable' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.synch_enable = match.group(1)
            else:
                print ("no matching key for " + key + " in " + str(object_handle))
                continue

    elif (object_handle.id == 'gtech_or2' or object_handle.id == 'gtech_and2' or object_handle.id == 'gtech_xor2'):
        for key, rx in rx_dict_AND2.items():
            match = rx.search(line)
            if (key == 'A' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.A = match.group(1)
            elif (key == 'B' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.B = match.group(1)
            elif (key == 'Z' and match):
                process_match([match.group(1)], object_handle, 'output', key)
                object_handle.Z = match.group(1)
            else:
                print ("no matching key for " + key + " in " + str(object_handle))

    elif (object_handle.id == 'gtech_not' or object_handle.id == 'gtech_buf'):
        for key, rx in rx_dict_BUF.items():
            match = rx.search(line)
            #print(match)
            #print(match.group(1))
            if (key == 'A' and match):
                process_match([match.group(1)], object_handle, 'input', key)
                object_handle.A = match.group(1)
            elif (key == 'Z' and match):
                process_match([match.group(1)], object_handle, 'output', key)
                object_handle.Z = match.group(1)
            else:
                print ("no matching key for " + key + " in " + str(object_handle))  

    elif (object_handle.id == 'mux_op'):
        for key, rx in rx_dict_MUX.items():
            match = rx.findall(line)
            if (key == 'D' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input', key)
                object_handle.datawidth = datawidth
                object_handle.D = matchlist
                for i in range(0,len(match)):
                    object_handle.D[i] = matchlist
            elif (key == 'S' and match):
                process_match(match, object_handle, 'input', key) # control 
                object_handle.S = np.append(object_handle.S, match)
            elif (key == 'Z' and match):
                process_match(match, object_handle, 'output', key)
                object_handle.Z = match
            else: 
                print ("no matching key for " + key + " in " + str(object_handle)) 

    elif (object_handle.id == 'select_op'):
        for key, rx in rx_dict_SELECT.items():
            match = rx.findall(line)
            if (key == 'DATA' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input', key)
                object_handle.datawidth = datawidth
                object_handle.D = matchlist
            elif (key == 'CONTROL' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input',key) # control
                object_handle.datawidth = datawidth
            elif (key == 'Z' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'output', key)
                object_handle.Z = matchlist
            else:
                print ("no matching key for " + key + " in " + str(object_handle)) 

    elif (object_handle.id == 'comp_op' or object_handle.id == 'add_op' or object_handle.id == 'sub_op' or object_handle.id == "mult_op" or object_handle.id == "div_op"):
        for key, rx in rx_dict_SUB_ADD_MULT.items():
            match = rx.findall(line)
            if (key == 'A' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input', key)
                object_handle.a_width = datawidth
                object_handle.A = matchlist
            elif (key == 'B' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input', key) # control
                object_handle.b_width = datawidth
                object_handle.B = matchlist
            elif (key == 'Z' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'output', key)
                object_handle.z_width = datawidth
                object_handle.Z = matchlist
            else: 
                print ("no matching key for " + key + " in " + str(object_handle))  

    elif (object_handle.id == 'shift_op' or object_handle.id == 'b_shift_op'):
        for key, rx in rx_dict_shift.items():
            match = rx.findall(line)
            if (key == 'A' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input', key)
                object_handle.a_width = datawidth
                object_handle.A = matchlist
            elif (key == 'SH' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input', key)  # control ?
                object_handle.sh_width = datawidth
                object_handle.SH = matchlist
            elif (key == 'Z' and match):
                dataN, datawidth, matchlist = process_match(match, object_handle, 'output', key)
                object_handle.z_width = datawidth
                object_handle.Z = matchlist
            else: 
                print ("no matching key for " + key + " in " + str(object_handle))   

    elif(object_handle.id == 'dep'):
        #look for dep objects and add them to list
        for key, rx in rx_dict_dep_internals.items():
            match = rx.findall(line)
            if match:
                object_handle.add_connections(match)

    elif(object_handle.id == 'assign'):
        #print("Assign statement")
        for key, rx in rx_dict_assign.items():
            match = rx.search(line)
            if key == 'rhs':
                rhsline = match.group(1)
                rhsline = rhsline.translate({ord(i): None for i in '}{\ '})
                if rhsline == "1'b0" or rhsline == "1'b1":
                    object_handle.rhs = "constant"
                else: 
                    i1, i2, new_rhsline, indextype = find_indexes(rhsline)
                    object_handle.rhs_i1 = i1
                    #look for new_rhsline in connections.
                    if (i2 != -1): object_handle.rhs_i2 = i2

                    element, foundbool = search_list(outputs, new_rhsline)
                    
                    if foundbool == False:
                        element, foundbool = search_list(inputs, new_rhsline)
                        if foundbool == False:
                            element, foundbool = search_list(connects, new_rhsline)
                    if foundbool:
                        object_handle.rhs = element
                    else:
                        print("Did not find match of rhs in assign")
                        print(new_rhsline)
    elif object_handle.id == 'unknown':
        for key, rx in rx_dict_unknown.items():
            #print(line)
            match = rx.findall(line)
            if key == 'S':
                dataN, datawidth, matchlist = process_match(match, object_handle, 'input', 'S')
                object_handle.datawidth = datawidth
                object_handle.S = matchlist
            elif key == 'Z':
                dataN, datawidth, matchlist = process_match(match, object_handle, 'output', 'Z')
                object_handle.datawidth = datawidth
                object_handle.S = matchlist
    
    else: 
        print (" No match found for object handle id: "+ object_handle.id)

    return key, match

# if signal is not constant, find out what it is connected to and the width and register connection
def process_match(match, object_handle, connection_type, port_name):


    #print(match)
    #print("Running process match for object " + str(object_handle) + " port " + port_name) 
    dataN = len(match)
    processed_matchlist = []

    if match[0] == "":
        return 0, 0, []

    for i in range(0, len(match)):

        match[i] = match[i].translate({ord(i): None for i in '}{\ '}) 
        datawidth = match[i].count(',')+1
        matchlist = match[i].split(',')
        processed_matchlist.append(matchlist)

        datawidth_set = {'sub_op','select_op', 'mux_op', 'shift_op', 'add_op', 'mult_op', 'comp_op', 'div_op', 'b_shift_op', 'shift_add_op'}
        if(connection_type == 'output'):
            if (object_handle.id in datawidth_set):
                if(object_handle.output_nodes ==[]):
                    object_handle.output_nodes = [None]*datawidth

        j_increment = 0
        for j in range(0, len(matchlist)): # j is i1
            
            if matchlist[j] == '1\'b1' or matchlist[j] == '1\'b0':
                datawidth = 1
            else:
                found = False
                if (port_name == 'dep'):
                    to_append = [object_handle, port_name, object_handle.id, connection_type, j+j_increment, i] #0]
                else:
                    to_append = [object_handle, port_name, object_handle.id, connection_type, j+j_increment]

                if (connection_type == 'input' or connection_type == 'control'):
                    i1, i2, matchobj, index_type = find_indexes(matchlist[j])
                    element, found = search_list(inputs, matchobj)
                    if found:
                        element.add_node_input_connection(i1, i2, to_append, index_type)
                    else:
                        element, found = search_list(connects, matchobj)
                        if found:
                            element.add_node_input_connection(i1, i2, to_append, index_type)
                        else: 
                            element, found = search_list(outputs, matchobj)
                            if found:
                                element.add_node_input_connection(i1, i2, to_append, index_type)
                            else:
                                print("Did not find: "+matchobj)
                                print("["+str(object_handle.name)+" , "+str(port_name)+" , "+str(object_handle.id)+"]")
                    if found:
                        if index_type == '':
                            #whole signal width-1 added to j
                            j_increment = j_increment + element.width -1
                        elif index_type == 'slice':
                            #add width of slice to j'
                            j_increment = j_increment+ i1-i2
                        #print("j_increment: "+str(j_increment))

                elif (connection_type == 'output'):
                    i1, i2, matchobj, index_type = find_indexes(matchlist[j])
                    element, found = search_list(outputs, matchobj)
                    if found:
                        if(i1 == -1 and i2 == -1):
                            #print("Element width: "+str(element.width))
                            if element.width > 1 and (element.width != len(object_handle.output_nodes)):
                            #redefine output
                                if (object_handle.id in datawidth_set):
                                    for i in range(element.width-1):
                                        object_handle.output_nodes.append(None)
                                    #print(len(object_handle.output_nodes))
                        if element.width == 1 and index_type == '':
                            index_type = 'bit'
                        element.add_node_output_connection(i1, i2, to_append, index_type,j+j_increment)
                    else:
                        element, found = search_list(connects, matchobj)
                        if found:
                            if(i1 == -1 and i2 == -1):
                                #print("Element width: "+str(element.width))
                                if element.width > 1 and (element.width != len(object_handle.output_nodes)):
                                #redefine output
                                    if (object_handle.id in datawidth_set):
                                        for i in range(element.width-1):
                                            object_handle.output_nodes.append(None)                            
            
                            if element.width == 1 and index_type == '':
                                index_type = 'bit'
                            #print("calling add_node_output_connection i1="+str(i1)+" i2: "+str(i2)+" j: "+str(j))
                            #print(index_type)
                            #print(j+j_increment)
                            element.add_node_output_connection(i1, i2, to_append, index_type,j+j_increment)
                        else: 
                            print("Did not find: "+matchobj)
                            print(str(object_handle)+" "+str(port_name)+" "+str(object_handle.id))
                    if found:
                        #print("j = "+str(j))
                        if index_type == '':
                            #whole signal width-1 added to j
                            #print("Modified j_increment")
                            j_increment = j_increment + element.width -1
                        elif index_type == 'slice':
                            #add width of slice to j'
                            #print("Modified j_increment")
                            j_increment = j_increment+ i1-i2
                        #print("j_increment: "+str(j_increment))
                else:
                    print("Did not find\t"+str(matchlist[j])+"\tANYWHERE")
        
    #print("\nFinished matchlist for: "+object_handle.name+"\nwidth: "+str(datawidth)+"\nN:" +str(dataN)+"\nMatchlist:" + str(processed_matchlist)+"\n")
    return dataN, datawidth,  processed_matchlist

#process module instantiation
def process_dependencies():#

    global modules

    for top_module in modules:# 
        set_global_lists(top_module)

        for dep in top_module.dependencies:# 
            found_dep = False
            #print("Looking for "+dep.modulename+" in dependencies")
            modulename = dep.modulename #
            #find modulename in module list
            
            for dep_module in modules:# 
                
                if dep_module.name == modulename:# 
                    found_dep = True
                    dep.module_handle = dep_module# 
                    module_connection_list = dep_module.connection_points
                    for dependency_connection_tuple in dep.connections[0]:#
                   
                        cleaned_dependency_connection_point = dependency_connection_tuple[0].translate({ord(i): None for i in '\ '})
                        found = False
                        
                        for module_connection_tuple in module_connection_list: 

                            if cleaned_dependency_connection_point == module_connection_tuple[0]:
                                found = True
                                cleaned_dep_connectionlist = dependency_connection_tuple[1].translate({ord(i): None for i in '}{\ '}) 
                                cleaned_dep_connectionlist = cleaned_dep_connectionlist.split(',')
                                
                                for i in range(len(cleaned_dep_connectionlist)):
                                    if len(module_connection_tuple) > 1:
                                        i1, i2, module_connection, typeindex = find_indexes(module_connection_tuple[1][i])
                                    else: 
                                        i1, i2, module_connection, typeindex = find_indexes(module_connection_tuple[0])

                                    dep_handle, found_dep = search_list(dep_module.inputs, module_connection)

                                    if found_dep:       
                                        print("sending "+str(cleaned_dep_connectionlist[i])+" into process match")
                                        dataN, datawidth, processed_matchlist = process_match([cleaned_dep_connectionlist[i]], dep_handle, 'input', 'dep' )
                                    else: 
                                        dep_handle, found_dep = search_list(dep_module.outputs, module_connection)
                                        #print(dep_connection)
                                        if found_dep:
                                            
                                            dataN, datawidth, processed_matchlist = process_match([cleaned_dep_connectionlist[i]], dep_handle, 'input', 'dep')
                                        else: 
                                            
                                            print("Did not find dep "+dep_module.name)
                                           
                        
                        #DEP NOT FOUND IN MODULES, MAYBE HINST
                        if found == False:
                            dep.possible_HINST = True

            if found_dep == False:
                print("Did not find dep: "+dep.modulename)                  
        top_module.set_lists()
        empty_global_lists()
                #find dep ports in inputs or outputs

#go structure heads and make structure trees
def connect_structure(module):

    global modules
    global top_level_parents

    for inp in module.inputs: # input of top module
        #print("processing input: " + inp.name)
        for nl in range(len(inp.connection_nodes)):
            structure_handle = structure(None, inp, nl)
            top_level_parents.append(structure_handle)
            top_level_parents_name.append(structure_handle.represented_object_handle.longname)
            #for n in range(len(inp.connection_nodes[nl])):
                #connect_children(structure_handle,nl,n)
        
    for m in modules: 

        for reg in m.regs: # reg of all modules

            #print("processing reg: " + reg.name)
            structure_handle = structure(None, reg, 0)
            top_level_parents.append(structure_handle)
            top_level_parents_name.append(structure_handle.represented_object_handle.longname)
            #connect_children(structure_handle, 0, 0)

    #print(len(structures))

    for s in top_level_parents:
        if s.structure_type == 'input':
            for n in range(len(s.represented_object_handle.connection_nodes[s.i1])):
                connect_children(s,nl,n)
        else:
            connect_children(s, 0, 0)

    global level
    global gtech_stack
    for s in top_level_parents:
        
        reg_structures_stack = []
        for key in type_dict:
            gtech_stack[key] = []
        level = 0

        add_real_intree(s)
        add_intree(s)
        

    #for s in top_level_parents:
    #    print(len(s.reg_children))
    
    #print(len(structures))

def add_intree(s):

    s.intree = s.intree + 1
    s.intree_count = s.intree_count + 1
    s_children = children_replace(s.children) 
    
    for child in s_children:         
        add_intree(child)


def add_real_intree(s):

    global level
    level = level + 1 
    global top_level_parents
    global reg_structures_stack
    global gtech_stack

    if s in top_level_parents:
        print(f"add intree:{s.represented_object_handle.longname}")
        reg_structures_stack = s

    s_real_children = real_children_replace(s.children) 

    for child in s_real_children:

        reg_structures_stack.gtech_real_num[child.represented_object_handle.id] += 1
        reg_structures_stack.gtech_real_depth[child.represented_object_handle.id] += level
        
        gtech_stack[child.represented_object_handle.id].append(child)         

        add_real_intree(child)

    level = level - 1


def real_children_replace(s_real_children):
    
    global gtech_stack
    global reg_structures_stack

    s_real_children_temp = []

    for child in s_real_children:

        if child.represented_object_handle.id in type_dict:
            if child not in gtech_stack[child.represented_object_handle.id]:
                s_real_children_temp.append(child)

        if child.represented_object_handle.id == 'reg':
            child.add_reg_parent(reg_structures_stack)
            reg_structures_stack.add_reg_child(child)

        elif child.represented_object_handle.id == 'input' or child.represented_object_handle.id == 'unknown':
            s_real_children_temp.extend(real_children_replace(child.children))

                

    return s_real_children_temp


def sub_intree(s):
    s.intree_count = s.intree_count - 1

    if s.intree_count == 0 and s.represented_object_handle.id != 'input' and s.represented_object_handle.id != 'reg':
        s.leaf_type.clear()
        s.width.clear()
        s.depth.clear()
        s.leaf_type_intree.clear()
        s.width_intree.clear()
        s.depth_intree.clear()
        s.feature_detected = False

    s_children = children_replace(s.children)
    for child in s_children:
        sub_intree(child)

#recursively connects all children to a parent and expand structure tree
def connect_children(parent, i1,i2):
    
    global structures
    global top_level_parents
    global structures_name

    object_handle = parent.represented_object_handle
    structures_name.append(object_handle.longname)

    #if parent in top_level_parents: # and object_handle.intree == 1:
    #    print(object_handle.intree)
    #return

    print("processing: " + object_handle.longname)
    #print(object_handle.output_nodes)
    
    i1 = parent.i1

    #if object_handle.id != 'reg' and object_handle.id != 'input' and object_handle.id != 'output':
    #object_handle.intree = object_handle.intree + 1


    if object_handle.id != 'input' and object_handle.id != 'output':
        
        datawidth_set = {'sub_op','select_op', 'mux_op', 'shift_op', 'add_op', 'mult_op', 'comp_op', 'div_op', 'b_shift_op', 'shift_add_op'}

        object_handle.structurecount = object_handle.structurecount+1

    #print("Current object: "+object_handle.name)
    output_nodes = []
    output_nodes_q = []
    output_nodes_qn = []
    #print("parent: "+object_handle.name)
    if object_handle.id == 'output':
        output_nodes = []
    elif object_handle.id == 'reg':

        #if object_handle.output_structure_taken == False:
        output_nodes_q = object_handle.output_nodes_q
        output_nodes_qn = object_handle.output_nodes_qn
            #object_handle.output_structure_taken = True
        
        if parent != None:# and parent.connected_inputs[0] != None:
            if parent.parents != []:
                object_handle.has_parent = True
        
    elif object_handle.id == 'input':

        output_nodes.append(object_handle.connection_nodes[i1][i2])

    elif object_handle.id == 'comp_op':
        output_nodes = object_handle.output_nodes

    elif object_handle.id == 'unknown':
        output_nodes = object_handle.output_nodes
        #print(output_nodes)

    else:
        if(len(object_handle.output_nodes)>i1):
            output_nodes = [object_handle.output_nodes[i1]]
        #ADDED to shorten recursion
            #object_handle.output_nodes[i1] = parent
    
    for node in output_nodes:
        if node != None:
            #if node.id == 'structure':
            #    parent = node
            #    print("end processing 1")
            #    return
            #if(node != None):

            #print(node.connected_inputs)
            for con in node.connected_inputs:
                    
                #if con[2] == 'reg' and con[2] != 'control':
                    #for s in top_level_parents:
                        #if con[0].name == s.represented_object_handle.name:
                            #print("aaaaaaaaaaaaaaaaaaaaa") 
                            #child_handle = s.represented_object_handle
                #else:
                child_handle = con[0]

                if (child_handle.longname in top_level_parents_name): 
                    for s in top_level_parents:
                        if s.represented_object_handle.longname == child_handle.longname:
                            parent.add_child(s)
                            s.add_parent(parent)
                            break
                    continue

                elif (child_handle.longname in structures_name): 
                    #if child_handle in top_level_parents:
                    #    continue
                    #else:
                    for s in structures:
                        if s.represented_object_handle.longname == child_handle.longname:
                            #print()
                            parent.add_child(s)
                            s.add_parent(parent)
                            #add_intree(s)
                            #s.represented_object_handle.intree = s.represented_object_handle.intree + 1
                            break
                    continue
                else:
                    structure_handle = structure(parent, child_handle, con[4])
                    added = parent.add_child(structure_handle)

                if added:

                    structure_handle.structure_type = child_handle.id
                    structure_handle.structure_connection_characteristic = con[3]

                    if con[2] == 'reg' and con[2] != 'control':
                        con[0].has_parent = True

                    #if structure_handle.represented_object_handle.id != 'reg' and structure_handle.represented_object_handle.id != 'input':
                        #structure_handle.represented_object_handle.output_nodes[0] = structure_handle

                    if(con[3] != 'control' and con[2] != 'reg' and con[2] != 'input'):
                        connect_children(structure_handle, con[4],node.i2)

                    elif(con[2] == 'input'):
                        #if input i2 needs to be set correctly
                        connect_children(structure_handle,con[4],con[5])#node.i2)
        else:
            pass 


    for node in output_nodes_q:
        if(node != None):
            for con in node.connected_inputs:

                #if con[2] == 'reg' and con[2] != 'control':
                    #for s in top_level_parents:
                        #if con[0].name == s.represented_object_handle.name:
                            #print("aaaaaaaaaaaaaaaaaaaaa") 
                            #child_handle = s.represented_object_handle
                #else:
                child_handle = con[0]

                if (child_handle.longname in top_level_parents_name): 
                    for s in top_level_parents:
                        if s.represented_object_handle.longname == child_handle.longname:
                            parent.add_child(s)
                            s.add_parent(parent)
                            break
                    continue

                elif (child_handle.longname in structures_name): 
                    #if child_handle in top_level_parents:
                    #    continue
                    #else:
                    for s in structures:
                        if s.represented_object_handle.longname == child_handle.longname:
                            #print()
                            parent.add_child(s)
                            s.add_parent(parent)
                            #add_intree(s)
                            #s.represented_object_handle.intree = s.represented_object_handle.intree + 1
                            break
                    continue
                else:
                    structure_handle = structure(parent, child_handle, con[4])
                    added = parent.add_child(structure_handle)

                if added:

                    #print("mark")
                    #for s in and2:
                    #    print(s.output_nodes[0].id)

                    structure_handle.structure_type = child_handle.id
                    structure_handle.structure_connection_characteristic = con[3]
                    if con[2] == 'reg' and con[2] != 'control':
                        #print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
                        con[0].has_parent = True

                        #print("mark")
                        #for s in and2:
                        #    print(s.output_nodes[0].id)

                    #if structure_handle.represented_object_handle.id != 'reg' and structure_handle.represented_object_handle.id != 'input':
                        #structure_handle.represented_object_handle.output_nodes[0] = structure_handle


                    if(con[3] != 'control' and con[2] != 'reg' and con[2] != 'input'):
                        connect_children(structure_handle, con[4],node.i2)

                    elif(con[2] == 'input'):
                        connect_children(structure_handle,con[4],con[5])


    for node in output_nodes_qn:
        if(node != None):
            
            #if node.id == 'structure':
                #parent = node
                #print("end processing 2")
            #    return

            for con in node.connected_inputs:
                #if con[2] == 'reg' and con[2] != 'control':
                    #for s in top_level_parents:
                        #if con[0].name == s.represented_object_handle.name:
                            #print("aaaaaaaaaaaaaaaaaaaaa") 
                            #child_handle = s.represented_object_handle
                #else:

                child_handle = con[0]

                if (child_handle.longname in top_level_parents_name): 
                    for s in top_level_parents:
                        if s.represented_object_handle.longname == child_handle.longname:
                            parent.add_child(s)
                            s.add_parent(parent)
                            break
                    continue

                elif (child_handle.longname in structures_name): 
                    #if child_handle in top_level_parents:
                    #    continue
                    #else:
                    for s in structures:
                        if s.represented_object_handle.longname == child_handle.longname:
                            #print()
                            parent.add_child(s)
                            s.add_parent(parent)
                            #add_intree(s)
                            #s.represented_object_handle.intree = s.represented_object_handle.intree + 1
                            break
                    continue
                else:
                    structure_handle = structure(parent, child_handle, con[4])
                    added = parent.add_child(structure_handle)

                if added:
                    structure_handle.structure_type = child_handle.id
                    structure_handle.structure_connection_characteristic = con[3]
                    if con[2] == 'reg' and con[2] != 'control':
                        con[0].has_parent = True

                    #if structure_handle.represented_object_handle.id != 'reg' and structure_handle.represented_object_handle.id != 'input':
                        #structure_handle.represented_object_handle.output_nodes[0] = structure_handle

                    if(con[3] != 'control' and con[2] != 'reg' and con[2] != 'input'):
                        connect_children(structure_handle, con[4],node.i2)
                    elif(con[2] == 'input'):
                        connect_children(structure_handle,con[4],con[5])
    #print("end processing 3")

    #if object_handle.id == 'reg' or object_handle.id == 'input':
    #    if len(reg_structures_stack) > 1:
    #        n = len(reg_structures_stack)
    #        reg_structures_stack[n-2].add_reg_child(reg_structures_stack[n-1])
    #        reg_structures_stack[n-1].add_reg_parent(reg_structures_stack[n-2])
    #    reg_structures_stack.pop()


def find_module(line):
    module_start = re.compile(r"module\s+(\S+)\s?\(")
    match = module_start.search(line)
    if (match):
        return match.group(1), True
    else:
        return None, False
    
def find_endmodule(line):
    module_end = re.compile(r"endmodule")
    match = module_end.search(line)
    if (match):
        return True
    else:
        return False    

#set nodes involved in assign statements equal to each other
def connect_assigns():
    #print("RUNNING CONNECT ASSIGNS")
    #print(assigns)
    for a in assigns:
        #print(a.name)
        #print(a.lhs.name+" "+str(a.lhs))
        #print(a.rhs)
        #print(a.i1)
        #print(a.i2)
        if(a.i1 != -1):
            #print(a.lhs.connection_nodes)
            lhs_node = a.lhs.connection_nodes[a.i1-a.lhs.widthoffset][a.i2]
        else: 
            #print("should connect whole signal")
            pass
        if a.rhs != None and a.rhs !="constant":
            #print(a.rhs.connection_nodes)
            rhs_node = a.rhs.connection_nodes[a.rhs_i1-a.rhs.widthoffset][a.rhs_i2]
        #print("Defined two connection nodes")
        if (a.i1 != -1 and a.rhs != None):
            #Connect bits of single node
            if a.rhs == "constant":
                a.lhs.connection_nodes[a.i1][a.i2].constant = True
            else:
                n1 = a.lhs.connection_nodes[a.i1-a.lhs.widthoffset][a.i2]
                n2 = a.rhs.connection_nodes[a.rhs_i1-a.rhs.widthoffset][a.rhs_i2]
                connect_nodes(n1, n2)
        else:
            #connect whole node if a.rhs exists
            if a.rhs != None and a.rhs != "constant":
                #print("Connect whole signal")
                for i in range(len(a.lhs.connection_nodes)):
                    for j in range(len(a.lhs.connection_nodes[i])):
                        n1 = a.lhs.connection_nodes[i][j]
                        n2 = a.rhs.connection_nodes[i][j]
                        connect_nodes(n1,n2)

#set nodes equal to each other
def connect_nodes(n1, n2):
    if n1 == n2:
        return # 
    else:
        for i in range(len(n1.connected_inputs)):
            n1_con = n1.connected_inputs[i]
            found = False
            for j in range(len(n2.connected_inputs)):
                n2_con = n2.connected_inputs[j]
                n1.connected_inputs.append(n2_con)
            n2.connected_inputs.append(n1_con)
        if n1.connected_outputs == []:
            n1.connected_outputs = n2.connected_outputs
        elif n2.connected_outputs == []:
            n2.connected_outputs = n1.connected_outputs
        if n1.constant == True: n2.constant = True
        if n2.constant == True: n1.constant = True

#set global lists relating to a module environment
def set_global_lists(module):
    global regs      
    global nots      
    global bufs      
    global and2s     
    global or2s      
    global muxes     
    global selects   
    global connects  
    global inputs    
    global outputs   
    global dependencies
    global shifters
    global comparators
    global xor2s
    global multipliers
    global subtractors
    global b_shifters 
    global adders     
    global shift_adders
    global divisors  
    global assigns

    regs            = module.regs        
    nots            = module.nots        
    bufs            = module.bufs        
    and2s           = module.and2s       
    or2s            = module.or2s        
    muxes           = module.muxes       
    selects         = module.selects     
    connects        = module.connects    
    inputs          = module.inputs      
    outputs         = module.outputs     
    dependencies    = module.dependencies
    shifters        = module.shifters    
    comparators     = module.comparators 
    xor2s           = module.xor2s       
    multipliers     = module.multipliers 
    subtractors     = module.subtractors 
    b_shifters      = module.b_shifters  
    adders          = module.adders      
    shift_adders    = module.shift_adders
    divisors        = module.divisors
    assigns         = module.assigns

#empty global lists relating to a module environment
def empty_global_lists():
    global regs      
    global nots      
    global bufs      
    global and2s     
    global or2s      
    global muxes     
    global selects   
    global connects  
    global inputs    
    global outputs   
    global dependencies
    global shifters
    global comparators
    global xor2s
    global multipliers
    global subtractors
    global b_shifters 
    global adders     
    global shift_adders
    global divisors  
    global assigns

    regs        = np.array([])
    nots        = np.array([])
    bufs        = np.array([])
    and2s       = np.array([])
    or2s        = np.array([])
    muxes       = np.array([])
    selects     = np.array([])
    connects    = np.array([])
    inputs      = np.array([])
    outputs     = np.array([])
    dependencies = np.array([])
    shifters    = np.array([])
    comparators = np.array([])
    xor2s       = np.array([])
    multipliers = np.array([])
    subtractors = np.array([])
    b_shifters = np.array([])
    adders      = np.array([])
    shift_adders  = np.array([])
    divisors    = np.array([])
    assigns = np.array([])

# create object
def make_object(line, name):
    #print("Making object: "+name)
    
    line = line.translate({ord(i): None for i in '}{\ '}) 

    if name == 'unknown_object':
        object_handle = unknown(line)
        object_handle.name = line
        return object_handle

    i1, i2, object_name, index_type = find_indexes(line)
    if ((i1 != -1) or (i2 != -1)) and name != 'register' and name != 'assign':
        object_handle = None
        foundbool = False
        if name == 'input':
            object_handle, foundbool = search_list(inputs, object_name)
        elif name == 'output':
            object_handle, foundbool = search_list(outputs, object_name)
        elif name == 'wire':
            object_handle, foundbool = search_list(connects, object_name)
        if (foundbool):
            #print("Found object in already existing connetion object")
            if (object_handle.width <= i1): object_handle.width = i1+1
            if (object_handle.depth <= i2): object_handle.depth = i2+1
        else:
            object_handle = create_object(name)
            reg_name = re.sub(r'_reg(?=\[)|_reg$','',object_name)
            object_handle.longname = f'{modulename_parsing}/{reg_name}'
            object_handle.name = reg_name
            object_handle.module_belong = modulename_parsing
            if i1 != -1: object_handle.width = i1+1
            if i2 != -1: object_handle.depth = i2+1
            object_handle.init_connection_nodes()
    else:
        object_handle = create_object(name)
        reg_name = re.sub(r'_reg(?=\[)|_reg$','',line)
        object_handle.longname = f'{modulename_parsing}/{reg_name}'
        object_handle.name = reg_name
        object_handle.module_belong = modulename_parsing
        if(name == 'input' or name == 'output' or name == 'wire'):
            object_handle.init_connection_nodes()
        if(name == 'assign'):
            #print("making fancy assign")
            object_handle.i1 = i1
            if i2 != -1: object_handle.i2 = i2
            foundbool = False
            connected_handle, foundbool = search_list(outputs, object_name)
            if foundbool != True:
                connected_handle, foundbool = search_list(inputs, object_name)
                if foundbool != True: 
                    connected_handle, foundbool = search_list(connects, object_name)
            if foundbool:
                object_handle.lhs = connected_handle
                
    return object_handle

#looks for indexes at end of string, returns i1, i2, str(w/o)indexes
def find_indexes(string):
    i1 = -1
    i2 = -1
    index_type = ''
    indexes = re.compile(r"(?:\[(\d{1,4})\])(?:\[(\d{1,4})\])?$")#
    slices = re.compile(r"(?:\[(\d{1,4})\:(\d{1,4})\])$")# 
    indexfind = indexes.search(string)
    newline = indexes.sub("",string)
    if indexfind != None:
        index_type  = 'index'
        if indexfind.group(1) != None:
            i1 = int(indexfind.group(1))
            if indexfind.group(2) != None:
                i2 = int(indexfind.group(2))
    else:
        slicefind = slices.search(string)
        if slicefind != None:
            index_type = 'slice'
            i1 = int(slicefind.group(1))
            i2 = int(slicefind.group(2))
            newline = slices.sub("", string)
            #print("found a slice")
    return i1, i2, newline, index_type

#create an object of a class specified by objectname
def create_object(objectname):
    objectselect = {
        'wire'      : connection,
        'input'     : input_obj,
        'output'    : output_obj,
        'SELECT_OP' : select_op,
        'MUX_OP'    : mux_op,
        'GTECH_NOT' : gtech_not,
        'GTECH_BUF' : gtech_buf,
        'GTECH_AND2': gtech_and2,
        'GTECH_OR2' : gtech_or2,
        'GTECH_XOR2': gtech_xor2,
        'register'  : register,
        'dep'       : dependency,
        'COMP_OP'   : comp_op,
        'SHIFT_OP'  : shift_op,
        'SUB_OP'    : sub_op,
        'ADD_OP'    : add_op,
        'MULT_OP'   : mult_op,
        'DIV_OP'    : div_op,
        'B_SHIFT_OP'   : b_shift_op,
        'SHIFT_ADD_OP': shift_add_op,
        'DIV_OP'    : div_op,
        'assign'    : assign
        
    }
    #get function
    #print("making object: "+objectname)
    func = objectselect.get(objectname, lambda: None)
    if func == None:
        print("found no object with objectname: "+str(objectname))
        return None
    else:
        #print("lookup successful, func = "+ str(func))
        retval = func()
        return retval

#look for object with name objectname in a list of objects. return handle if match, None otherwise
def find_object(objectlist, objectname):
    for i in range(0, len(objectlist)-1):
        if objectlist[i].name == objectname:
            return objectlist[i]
    return None

#return object in list with name matching searchstring or None, False
def search_list(list, searchstring):
    element = None
    for element in list:
        if (element.name == searchstring):
            #if searchstring[0] == 'e': print("FOUND: "+searchstring)
            return element, True
    return element, False

#count gates in representation
def count_gates(module):
    global reg_n          
    global not_n      
    global buf_n        
    global and2_n        
    global or2_n         
    global mux_n         
    global select_n          
    global shift_n      
    global comp_n     
    global xor2_n         
    global mult_n     
    global sub_n      
    global b_shift_n    
    global add_n        
    global shift_add_n  
    global div_n    
    global input_n   
    m = module  
    if m != None:
        for r in m.regs:
            #if r.output_nodes_q[0] != None or r.output_nodes_qn[0] != None:
                #if r.has_parent:
                #print(r.name) 
            reg_n       =  reg_n +1
                #else:
                   # reg_n = reg_n+1
        not_n       =  not_n           + len(m.nots        )
        buf_n       =  buf_n           + len(m.bufs        )
        and2_n      =  and2_n          + len(m.and2s       )
        or2_n       =  or2_n           + len(m.or2s        )
        mux_n       =  mux_n           + len(m.muxes       )
        select_n    =  select_n        + len(m.selects     )
        shift_n     =  shift_n         + len(m.shifters    )
        comp_n      =  comp_n          + len(m.comparators )
        xor2_n      =  xor2_n          + len(m.xor2s       )
        mult_n      =  mult_n          + len(m.multipliers )
        sub_n       =  sub_n           + len(m.subtractors )
        b_shift_n   =  b_shift_n       + len(m.b_shifters  )
        add_n       =  add_n           + len(m.adders      )
        shift_add_n =  shift_add_n     + len(m.shift_adders)
        div_n       =  div_n           + len(m.divisors    )
        input_n     =  input_n         + len(m.inputs      )
        for d in module.dependencies:
            m = d.module_handle
            count_gates(m)

#print gate counts
def print_gates():
    print("regs\t\t"+ str(reg_n      )) 
    print("muxes\t\t"+ str(mux_n      )) 
    print("nots\t\t"+ str(not_n      )) 
    print("bufs\t\t"+ str(buf_n      ))     
    print("arithmetic:\t"+str(mult_n+sub_n+add_n+shift_add_n+div_n))
    print("logic:\t\t"+str(and2_n+or2_n+shift_n+comp_n+xor2_n+b_shift_n))
    print("selects\t\t"+ str(select_n   )) 
    print("Total:\t\t "+str(reg_n+not_n+buf_n+and2_n+or2_n+mux_n+select_n+shift_n+comp_n+xor2_n+mult_n+sub_n+b_shift_n+add_n+shift_add_n+div_n))
    print("inputs\t\t"+ str(input_n      )) 
    print()

#print name of all objects in a list
def print_list_names(l):
    for e in l:
        print("\t"+e.name) 

#############################################################################################################################
#                                                        classes                                                          
#############################################################################################################################

class unknown:
    id = 'unknown'
    S = np.array([])
    Z = 0
    datawidth = 0
    def __init__(self,name):
        global unknowns
        self.output_nodes = [None]*1
        unknowns = np.append(unknowns, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

    def __str__(self):
        return "unknown: " + self.name

class module:
    name = ""
    connection_point_string = ""
    def __init__(self, name):
        self.regs           = []
        self.nots           = []
        self.bufs           = []
        self.and2s          = []
        self.or2s           = []
        self.muxes          = []
        self.selects        = []
        self.connects       = []
        self.inputs         = []
        self.outputs        = []
        self.dependencies   = []
        self.shifters       = []
        self.comparators    = []
        self.xor2s          = []
        self.multipliers    = []
        self.subtractors    = []
        self.b_shifters     = []
        self.adders         = []
        self.shift_adders   = []
        self.divisors       = []
        self.assigns        = []
        self.name = name
        self.connection_points = []
    def set_lists(self):
        global regs      
        global nots      
        global bufs      
        global and2s     
        global or2s      
        global muxes     
        global selects   
        global connects  
        global inputs    
        global outputs   
        global dependencies
        global shifters
        global comparators
        global xor2s
        global multipliers
        global subtractors
        global b_shifters 
        global adders     
        global shift_adders
        global divisors  
        global assigns

        self.regs           = np.copy(regs        )
        self.nots           = np.copy(nots        )
        self.bufs           = np.copy(bufs        )
        self.and2s          = np.copy(and2s       )
        self.or2s           = np.copy(or2s        )
        self.muxes          = np.copy(muxes       )
        self.selects        = np.copy(selects     )
        self.connects       = np.copy(connects    )
        self.inputs         = np.copy(inputs      )
        self.outputs        = np.copy(outputs     )
        self.dependencies   = np.copy(dependencies)
        self.shifters       = np.copy(shifters    )
        self.comparators    = np.copy(comparators )
        self.xor2s          = np.copy(xor2s       )
        self.multipliers    = np.copy(multipliers )
        self.subtractors    = np.copy(subtractors )
        self.b_shifters     = np.copy(b_shifters  )
        self.adders         = np.copy(adders      )
        self.shift_adders   = np.copy(shift_adders)
        self.divisors       = np.copy(divisors    )
        self.assigns        = np.copy(assigns     )
    def set_connection_points(self):
        self.connection_point_string = self.connection_point_string.translate({ord(i): None for i in '\ \n'}) 
        #print(self.connection_point_string)
        connectionpoints = []
        for key, rx in rx_dict_module_connections.items():
            match = rx.findall(self.connection_point_string) #rx.findall(line)
            #print("found "+str(len(match))+" matches in modulestring")
            if match:
                #print(str(match))
                for m in match:
                    if (m[0] == ''): connectionpoints.append(tuple([m[2]]))
                    else:
                        l =  m[1].translate({ord(i): None for i in '}{'}) 
                        l = l.split(',')
                        connectionpoints.append(tuple([m[0], l]))
                #connectionpoints.append(match)
        #print(connectionpoints)
        self.connection_points = connectionpoints

class register:
    id = 'reg'
    clear = 0
    preset = 0 
    next_state = 0
    clocked_on = 0
    data_in = 0
    enable = 0
    Q = 0
    QN = 0
    synch_clear = 0
    synch_preset = 0
    synch_toggle = 0
    synch_enable = 0
    output_structure_taken = False
    def __str__(self):
        #return "Register: \n clear = " + str(self.clear) + "\n preset = " + str(self.preset) + 
        #"\n next_state = " + str(self.next_state) +"\n clocked_on = " + str(self.clocked_on) +"\n data_in = 
        #" + str(self.data_in) +"\n enable = " + str(self.enable) +"\n Q = " + str(self.Q)
        return "Register: " + self.name
    def __init__(self):
        #print("made reg!")
        global regs
        regs = np.append(regs, self)
        self.output_nodes_q = [None]*1#���*1��֪���Ǹ�ɶ�ģ���Ī������
        self.output_nodes_qn = [None]*1
        self.has_parent = False

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

class gtech_or2:
    id = 'gtech_or2'
    A = 0
    B = 0
    Z = 0
    def __str__(self):
        return "OR2: " + self.name
    def __init__(self):
        #print("made or2!")
        global or2s
        or2s = np.append(or2s, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = [None]*1

class gtech_and2:
    global and2
    id = 'gtech_and2'
    A = 0
    B = 0
    Z = 0
    def __str__(self):
        
        return "AND2: " + self.name
        #return "AND2: " + self.name + " A: "+str(self.A)+" B: "+str(self.B)+" Z: "+str(self.Z)
    def __init__(self):
        #and2.append(self)
        global and2s
        and2s = np.append(and2s, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = [None]*1

class gtech_xor2:
    id = 'gtech_xor2'
    A = 0
    B = 0
    Z = 0
    def __str__(self):
        return "XOR2: " + self.name
    def __init__(self):
        #print("made or2!")
        global xor2s
        xor2s = np.append(xor2s, self)   

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = [None]*1

class gtech_not:
    id = 'gtech_not'
    A = 0
    Z = 0
    def __str__(self):
        return "NOT: " + self.name
    def __init__(self):
        global nots
        self.output_nodes = [None]*1
        nots = np.append(nots, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

class gtech_buf:
    id = 'gtech_buf'
    A = 0
    Z = 0
    def __str__(self):
        return "BUF: " + self.name
    def __init__(self):
        self.output_nodes = [None]*1
        global bufs
        bufs = np.append(bufs, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

class connection:
    id = 'connection'
    width       = 1
    depth       = 1
    widthoffset = 0
    def __init__(self):
        global connects
        connects = np.append(connects, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.connection_nodes = []
        self.init_connection_nodes()
    def init_connection_nodes(self):
        self.connection_nodes = [[node(j,i) for i in range(self.depth)] for j in range(self.width)] 
    def add_node_input_connection(self,i1, i2, l, index_type):
        #to increment l[4]
        add = 0
        if index_type == '':
            for i in range(self.width):
                li = l[:]
                li[4] = l[4] + add
                add = add+1
                for j in range(self.depth):
                    self.connection_nodes[i][j].add_input_connection(li)
                
        elif index_type == 'index':
            if((i1-self.widthoffset)<len(self.connection_nodes)):
                self.connection_nodes[i1-self.widthoffset][i2].add_input_connection(l)
        elif index_type == 'slice':
            for i in range(i2, i1):
                li = l[:]
                li[4] = l[4] + add
                self.connection_nodes[i-self.widthoffset][0].add_input_connection(li)
                add = add+1
                
    def add_node_output_connection(self,i1, i2, l, index_type,k):
        add = 0
        if index_type == '':
            for i in range(self.width):
                li = l[:]
                li[4] = l[4] + add
                k = li[4]
                for j in range(self.depth):
                    self.connection_nodes[i][j].add_output_connection(li,k)
                add = add+1
        elif index_type == 'bit':
            self.connection_nodes[i1-self.widthoffset][i2].add_output_connection(l,k)
        
        elif index_type == 'index':
            if((i1-self.widthoffset)<len(self.connection_nodes)):
                self.connection_nodes[i1-self.widthoffset][i2].add_output_connection(l,k)
        elif index_type == 'slice':
            for i in range(i2, i1):
                li = l[:]
                li[4] = l[4] + add
                k = li[4]
                if(i<len(self.connection_nodes)):
                    self.connection_nodes[i][0].add_output_connection(li,k)
                add = add+1

class assign:
    id = 'assign'
    lhs = ''#
    rhs = ''
    i1 = 0
    i2 = 0
    lhs = None
    rhs = None #
    rhs_i1 = 0
    rhs_i2 = 0
    def __init__(self):   
        global assigns
        assigns = np.append(assigns,self) #
    #    self.lhs = lhs
    #    self.rhs = rhs

class node:
    id = 'node'
    def __init__(self,i1, i2):
        self.connected_inputs      = []
        self.connected_outputs     = []
        self.i1 = i1
        self.i2 = i2
        self.constant = False
    def add_input_connection(self,l):
        self.connected_inputs.append(l)
    def add_output_connection(self,l,j):
        self.connected_outputs.append(l)
        connected_object_handle = l[0]
        if l[2] != 'reg':
            if(j<len(connected_object_handle.output_nodes)):
            #add output node to connected object
                connected_object_handle.output_nodes[j] = self
        else: 
            #add output node to register
            connected_object_handle.output_nodes_q[0] = self
            connected_object_handle.output_nodes_qn[0] = self
    def print(self):
        print("Node:")
        print("\tinputs")
        print(self.connected_inputs)
        print("\toutputs:")
        print(self.connected_outputs)
        print("Endnode")

class input_obj():
    id      = 'input'
    width   = 1
    depth = 1
    widthoffset = 0
    def __init__(self):
        global inputs
        inputs = np.append(inputs, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.connection_nodes  = []
    def init_connection_nodes(self):
        #print("\ninitializing connection nodes")
        self.connection_nodes = [[node(j,i) for i in range(self.depth)] for j in range(self.width)] 
        #print(self.connection_nodes)
    def add_node_input_connection(self,i1, i2, l, index_type):
        add = 0
        if index_type == '':
            if (i1 == -1 and i2 == -1):
                for i in range(self.width):
                    li = l[:]
                    li[4] = l[4] + add
                    for j in range(self.depth):
                        #print("added input connection with l[4]: "+str(li[4]))
                        self.connection_nodes[i][j].add_input_connection(li)
                    add = add+1

        elif index_type == 'index':
            self.connection_nodes[i1-self.widthoffset][i2].add_input_connection(l)
        elif index_type == 'slice':
            for i in range(i2, i1):
                li = l[:]
                li[4] = l[4] + add
                self.connection_nodes[i][0].add_input_connection(li)
                add = add+1

class output_obj():
    id      = 'output'
    width   = 1
    depth = 1
    widthoffset = 0
    def __init__(self):
        global outputs
        outputs = np.append(outputs, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.connection_nodes = []
    def init_connection_nodes(self):
        #print("\ninitializing connection nodes")
        self.connection_nodes = [[node(j,i) for i in range(self.depth)] for j in range(self.width)] 
        #print(self.connection_nodes)
    def add_node_output_connection(self,i1, i2, l, index_type,k):
        add = 0
        if index_type == '':
            for i in range(self.width):
                li = l[:]
                li[4] = l[4] + add
                k = li[4]
                for j in range(self.depth):
                    self.connection_nodes[i][j].add_output_connection(li,k)
                add = add+1
                #print(add)
        elif index_type == 'bit':
            self.connection_nodes[i1-self.widthoffset][i2].add_output_connection(l,k)
        
        elif index_type == 'index':
            self.connection_nodes[i1-self.widthoffset][i2].add_output_connection(l,k)
        elif index_type == 'slice':
            for i in range(i2, i1):
                li = l[:]
                li[4] = l[4] + add
                k = li[4]
                self.connection_nodes[i][0].add_output_connection(li,k)
                add = add+1
    def add_node_input_connection(self,i1, i2, l, index_type):
        add = 0
        if index_type == '':
            if (i1 == -1 and i2 == -1):
                for i in range(self.width):
                    li = l[:]
                    li[4] = l[4] + add
                    for j in range(self.depth):
                        self.connection_nodes[i][j].add_input_connection(li)
                    add = add+1

        elif index_type == 'index':
            self.connection_nodes[i1-self.widthoffset][i2].add_input_connection(l)
        elif index_type == 'slice':
            for i in range(i2, i1):
                li = l[:]
                li[4] = l[4] + add
                self.connection_nodes[i][0].add_input_connection(li)
                add = add+1

class dependency: 
    #name of instantiation
    id = 'dep'
    #modulename
    modulename = ""
    module_handle = None
    possible_HINST = False
    def __init__(self):
        global dependencies
        dependencies = np.append(dependencies, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.connections = []
    def add_connections(self, list):
        self.connections.append(list)

class select_op: 
    #NB select also has width of select to take into account
    id = 'select_op'
    D        = np.array([])
    CONTROL     = np.array([])
    Z           = np.array([])
    datawidth   = 0
    selectwidth = 0
    def __init__(self):
        global selects
        self.output_nodes = []
        selects = np.append(selects, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

    def __str__(self):
        return "select: " + self.name + " inputs: \n" + str((self.D)) + " \nselect:\n" + str((self.CONTROL)) +"\n datawidth = " +str(self.datawidth)+ " selectwidth = "+ str(self.selectwidth)

class mux_op:
    id = 'mux_op'
    D = np.array([])
    S = np.array([])
    Z = np.array([])
    datawidth = 0
    def __init__(self):
        global muxes
        self.output_nodes = []
        muxes = np.append(muxes, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

    def __str__(self):
        return "mux: " + self.name + "# inputs: " + str(self.d_size()) + " datawidth: " + str(self.datawidth)
    def d_size(self):
        D = self.D
        print (str(D))
        #print("number of Ds " + str(len(D)))
        return len(D) #D.size()
    def s_size(self):
        S = self.S
        return len(S)
    #def datawidth(self):
    #    D = self.D
    #    return len(D.item(0))
    def print(self):
        print("mux: " + self.name + "# inputs: " + str(self.d_size()) + " datawidth: " + str(self.datawidth))

class comp_op:
    id = "comp_op"
    A = 0
    B = 0
    Z = 0
    a_width = 0
    b_width = 0
    z_width = 0
    def __init__(self):
        global comparators 
        comparators = np.append(comparators, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []
        #print("Made comparator")

class sub_op:
    id = "sub_op"
    A = 0
    B = 0
    Z = 0
    def __init__(self):
        global subtractors 
        subtractors = np.append(subtractors, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []
        #print("made subtractor")

class add_op:
    id = "add_op"
    A = 0
    B = 0
    Z = 0
    a_width = 0
    b_width = 0
    z_width = 0
    def __init__(self):
        global adders
        adders = np.append(adders, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []
        #print("made adder")

class mult_op:
    id = "mult_op"
    A = 0
    B = 0 
    Z = 0
    a_width = 0
    b_width = 0
    z_width = 0
    def __init__(self):
        global multipliers
        multipliers = np.append(multipliers, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []
        #print("made multiplicator")

class div_op:
    id = "div_op"
    A = 0
    B = 0 
    Z = 0
    a_width = 0
    b_width = 0
    z_width = 0
    def __init__(self):
        global divisors
        divisors = np.append(divisors, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []
        #print("Made divisor")

class shift_op:
    id = 'shift_op'
    A = 0
    SH = 0
    Z = 0
    a_width = 0
    sh_width = 0
    z_width = 0
    def __init__(self):
        global shifters
        shifters = np.append(shifters,self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []

class b_shift_op:
    id = "b_shift_op"
    A = 0 
    SH = 0
    Z = 0
    a_width = 0
    sh_width = 0
    z_width = 0
    def __init__(self):
        global b_shifters
        b_shifters = np.append(b_shifters, self)

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []
        #print("made barrelshift")

class shift_add_op:
    #dont know what content should be here, may need to implement as I go if I encounter it during testing
    id = "shift_add_op"

    def __init__(self):          
        
        global shift_adders
        shift_adders = np.append(shift_adders, self) 

        self.name = ""
        self.longname = ""
        self.structurecount = 0
        self.intree = 0
        self.module_belong = ''

        self.output_nodes = []
        #print("Made shift adder")

#structure tree class
class structure:
    global structures
    structure_type = ''
    structure_connection_characteristic = ''
    id = 'structure'

    def __init__(self, parent, represented_object_handle, i1):
        self.parents = []
        if parent != None:
            self.parents.append(parent)
        self.children = []
        self.reg_parents = []
        self.reg_children = []
        self.represented_object_handle = represented_object_handle
        self.i1 = i1
        self.powerStructure = None

        self.intree = 0
        self.intree_count = 0
        self.module_belong = ''

        self.feature_detected = False
        self.reg_feature_detected = False

        self.reg_width = np.array([])
        self.reg_depth = np.array([])
        self.reg_children_num = 0 # sum of all the children's children num
        self.reg_width_intree = np.array([])
        self.reg_depth_intree = np.array([])
        self.reg_children_num_intree = 0 # sum of all the children's children num

        self.crossing_tree = {
            "height"      : 0,
            "leafs"       : 0,
            "gtech_or2"   : 0,
            "gtech_and2"  : 0,
            "gtech_xor2"  : 0,
            "gtech_not"   : 0,
            "gtech_buf"   : 0,
            "select_op"   : 0,
            "mux_op"      : 0,
            "comp_op"     : 0,
            "sub_op"      : 0,
            "add_op"      : 0,
            "mult_op"     : 0,
            "div_op"      : 0,
            "shift_op"    : 0
        }


        self.gtech_real_num = {
            "gtech_or2"   : 0,
            "gtech_and2"  : 0,
            "gtech_xor2"  : 0,
            "gtech_not"   : 0,
            "gtech_buf"   : 0,
            "select_op"   : 0,
            "mux_op"      : 0,
            "comp_op"     : 0,
            "sub_op"      : 0,
            "add_op"      : 0,
            "mult_op"     : 0,
            "div_op"      : 0,
            "shift_op"    : 0
        }

        self.gtech_real_depth = {
            "gtech_or2"   : 0,
            "gtech_and2"  : 0,
            "gtech_xor2"  : 0,
            "gtech_not"   : 0,
            "gtech_buf"   : 0,
            "select_op"   : 0,
            "mux_op"      : 0,
            "comp_op"     : 0,
            "sub_op"      : 0,
            "add_op"      : 0,
            "mult_op"     : 0,
            "div_op"      : 0,
            "shift_op"    : 0
        }

        self.leaf_type = {
            # 'intree' type:
            # children's type of gtechs
            # example: gtech_or2_leaf_type = [1,2,0...]:
            #          the gtech_or2s have one child gtech_or2 two children gtech_and2s
            "gtech_or2"   : np.array([]),
            "gtech_and2"  : np.array([]),
            "gtech_xor2"  : np.array([]),
            "gtech_not"   : np.array([]),
            "gtech_buf"   : np.array([]),
            "select_op"   : np.array([]),
            "mux_op"      : np.array([]),
            "comp_op"     : np.array([]),
            "sub_op"      : np.array([]),
            "add_op"      : np.array([]),
            "mult_op"     : np.array([]),
            "div_op"      : np.array([]),
            "shift_op"    : np.array([])
        }

        self.leaf_type_intree = {
            "gtech_or2"   : np.array([]),
            "gtech_and2"  : np.array([]),
            "gtech_xor2"  : np.array([]),
            "gtech_not"   : np.array([]),
            "gtech_buf"   : np.array([]),
            "select_op"   : np.array([]),
            "mux_op"      : np.array([]),
            "comp_op"     : np.array([]),
            "sub_op"      : np.array([]),
            "add_op"      : np.array([]),
            "mult_op"     : np.array([]),
            "div_op"      : np.array([]),
            "shift_op"    : np.array([])
        }

        self.width = {
            # number of gtechs in different level
            # example: gtech_or2_width = [1,2,3,...]:
            #          one gtech_or2 in the 1st level below 
            #          two gtech_or2s in the 2nd level below 
            "gtech_or2"   : np.array([]),
            "gtech_and2"  : np.array([]),
            "gtech_xor2"  : np.array([]),
            "gtech_not"   : np.array([]),
            "gtech_buf"   : np.array([]),
            "select_op"   : np.array([]),
            "mux_op"      : np.array([]),
            "comp_op"     : np.array([]),
            "sub_op"      : np.array([]),
            "add_op"      : np.array([]),
            "mult_op"     : np.array([]),
            "div_op"      : np.array([]),
            "shift_op"    : np.array([])
        }

        self.width_intree = {
            "gtech_or2"   : np.array([]),
            "gtech_and2"  : np.array([]),
            "gtech_xor2"  : np.array([]),
            "gtech_not"   : np.array([]),
            "gtech_buf"   : np.array([]),
            "select_op"   : np.array([]),
            "mux_op"      : np.array([]),
            "comp_op"     : np.array([]),
            "sub_op"      : np.array([]),
            "add_op"      : np.array([]),
            "mult_op"     : np.array([]),
            "div_op"      : np.array([]),
            "shift_op"    : np.array([])
        }

        self.depth = {
            # number of gtechs in different path
            # example: gtech_or2_depth = [1,2,3,...]:
            #          one gtech_or2 in the 1st path below 
            #          two gtech_or2s in the 2nd path below
            "gtech_or2"   : np.array([]),
            "gtech_and2"  : np.array([]),
            "gtech_xor2"  : np.array([]),
            "gtech_not"   : np.array([]),
            "gtech_buf"   : np.array([]),
            "select_op"   : np.array([]),
            "mux_op"      : np.array([]),
            "comp_op"     : np.array([]),
            "sub_op"      : np.array([]),
            "add_op"      : np.array([]),
            "mult_op"     : np.array([]),
            "div_op"      : np.array([]),
            "shift_op"    : np.array([])
        }

        self.depth_intree = {
            "gtech_or2"   : np.array([]),
            "gtech_and2"  : np.array([]),
            "gtech_xor2"  : np.array([]),
            "gtech_not"   : np.array([]),
            "gtech_buf"   : np.array([]),
            "select_op"   : np.array([]),
            "mux_op"      : np.array([]),
            "comp_op"     : np.array([]),
            "sub_op"      : np.array([]),
            "add_op"      : np.array([]),
            "mult_op"     : np.array([]),
            "div_op"      : np.array([]),
            "shift_op"    : np.array([])
        }

        structures.append(self)

    def add_child(self, child):
        if child.represented_object_handle == self.represented_object_handle:
                return False 
        for c in self.children:
            if c.represented_object_handle == child.represented_object_handle:
                return False 
        self.children.append(child)
        return True
    def add_parent(self, parent):
        if parent.represented_object_handle == self.represented_object_handle:
            return False 
        for c in self.parents:
            if c.represented_object_handle == parent.represented_object_handle:
                return False 
        self.parents.append(parent)
        return True

    def add_reg_child(self, reg_child):
        if reg_child.represented_object_handle == self.represented_object_handle:
                return False 
        for c in self.reg_children:
            if c.represented_object_handle == reg_child.represented_object_handle:
                return False 
        self.reg_children.append(reg_child)
        return True
    def add_reg_parent(self, reg_parent):
        if reg_parent.represented_object_handle == self.represented_object_handle:
            return False 
        for c in self.reg_parents:
            if c.represented_object_handle == reg_parent.represented_object_handle:
                return False 
        self.reg_parents.append(reg_parent)
        return True


    def print(self):
        if self.children != []: print("{", end = '')
        for child in self.children:
            print(child.represented_object_handle.id+" ,",end = '')
            child.print()
        if self.children != []: print("}", end = '')

    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.represented_object_handle.longname+"/"+str(self.intree))+"\n"
        if level < 20:
            for child in self.children:
                if child.represented_object_handle.id == 'reg':
                    continue
                ret += child.__repr__(level+1)
        return ret


#dictionaries containing regular expressions to handle different constructs from the elaborated systemverilog
rx_dict_module_start = {
    'begin'     : re.compile(r"module\s+(\S+)")#module
}
rx_dict_objectconnection = {
    'bit'       : re.compile(r"(1'b\d)"),#
    'connect'   : re.compile(r"(\S+)(?:\[(\d{1,4})\])?(?:\[(\d{1,4})\])?")#
}#
rx_dict_start = { 
    'register'      : re.compile(r"\\\*\*SEQGEN\*\*\s+(\S+)\s" ), # SEQGEN
    'GTECH_OR2'     : re.compile(r"GTECH_OR2\s+(\S+)\s"), # GTECH_OR2
    'GTECH_NOT'     : re.compile(r"GTECH_NOT\s+(\S+)\s"), # GTECH_NOT
    'GTECH_BUF'     : re.compile(r"GTECH_BUF\s+(\S+)\s"), # GTECH_BUF
    'GTECH_AND2'    : re.compile(r"GTECH_AND2\s+(\S+)\s"), # GTECH_AND2
    'GTECH_XOR2'    : re.compile(r"GTECH_XOR2\s+(\S+)\s"), # GTECH_XOR2
    'MUX_OP'        : re.compile(r"MUX_OP\s+(\S+)\s"), # MUX_OP 
    'SELECT_OP'     : re.compile(r"SELECT_OP\s+(\S+)\s"), # SELECT_OP
    #TODO: mux add, sub,mult, shifts and compares remain at least...
    #all of these can be single bit or multibit. if square brackets before name
    #multi, else single (group capture?)
    'input'         : re.compile(r"(input)\s+(?:\[(\d{1,3}):(\d{1,3})\])?"), # 
    'output'        : re.compile(r"(output)\s+(?:\[(\d{1,3}):(\d{1,3})\])?"), #
    'wire'          : re.compile(r"(wire)\s+(?:\[(\d{1,3}):(\d{1,3})\])?"), 
    'COMP_OP'       : re.compile(r"^\s*(?:EQ_UNS_OP|NE_UNS_OP|EQ_TC_OP|NE_TC_OP|GEQ_UNS_OP|GEQ_TC_OP|LEQ_UNS_OP|LEQ_TC_OP|GT_UNS_OP|GT_TC_OP|LT_UNS_OP|LT_TC_OP)\s+(\S+)\s"), #
    'SUB_OP'        : re.compile(r"SUB_(?:UNS_OP|UNS_CI_OP|TC_OP|TC_CI_OP)\s+(\S+)\s"), #
    'ADD_OP'        : re.compile(r"ADD_(?:UNS_OP|UNS_CI_OP|TC_OP|TC_CI_OP)\s+(\S+)\s"), #
    'MULT_OP'       : re.compile(r"MULT_(?:UNS_OP|TC_OP)\s+(\S+)\s"), #
    'DIV_OP'        : re.compile(r"(?:DIV|MOD|REM|DIVREM|DIVMOD)_(?:UNS|TC)_OP\s+(\S+)\s"), #only div in Yoda
    'SHIFT_OP'      : re.compile(r"(?:ASH|ASHR|SRA)_(?:UNS|TC)_(?:UNS|TC|OP)(?:_OP)?\s+(\S+)\s"), 
    'B_SHIFT_OP'    : re.compile(r"BSH(?:_UNS_OP|_TC_OP|L_TC_OP|R_UNS_OP|R_TC_OP)\s+(\S+)\s"), #not in Yoda
    'SHIFT_ADD_OP'  : re.compile(r"(?:SLA_UNS_OP|SLA_TC_OP)\s+(\S+)\s"), #not in Yoda
    'assign'        : re.compile(r"assign\s([^=]+)"),
    #'dep'           : re.compile(r"(\S+)\s+(\S*u_\S+)\s*\("), #
    'dep'           : re.compile(r"(\S+)\s+(\S+)\s*\("),#
    #'SRA'           :
}
rx_dict_unknown = {
    'S'   : re.compile(r"\.\w+\(([^\)]*)\)\,"),
    'Z'         : re.compile(r"\..*\(([^\)]*)\)"),
    #'dep'   : re.compile(r"\s?(?:\.(?P<connection_point>[^\(\s,]+)\((?P<connected_to>[^\(\);\s]*)\),?)\s?")
}#
rx_dict_dep_internals = {
    #'dep'   : re.compile(r"\(\s?(?:\.([^\(\s,]+)\(([^\(\);\s]*)\),?)+\s?\);")
    'dep'   : re.compile(r"\s?(?:\.(?P<connection_point>[^\(\s,]+)\((?P<connected_to>[^\(\);\s]*)\),?)\s?")
}#Ѱ������.connection_point(connected_to)
rx_dict_assign = {
    'rhs' : re.compile(r"=([^=]+);")
}#���� = expression;
rx_dict_shift = {
    'A'     : re.compile(r"\.A\(([^\)]*)\)"),
    'SH'     : re.compile(r"\.SH\(([^\)]*)\)"),
    'Z'     : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_module_connections = {
    #'reconnect'   : re.compile(r"\s?((?:\.(?P<connection_point>[^\(\s,]+)\((?P<connected_to>[^\(\);\s]*)\)))[,\)]\s?")#|([^,\.\(\)\{\}\[\]]+)[,\)]\s?"),
    #'plain'       : re.compile(r"\s*([^\(\s,]+)\s?[,\)]")
    'connection' : re.compile(r"(?:[,\(]\.([^\(\),]+)\(([^\(\)]+)\))|(?:[,\(]([^\.][^\)\(,;]*))")
}
rx_dict_end = {
    #'end'   : re.compile(r"\);"),
    'semi'  : re.compile(r";") #hopefully this does not ruin anything and all semicolons are ends
}
rx_dict_in_out_wire = {
    'varname' : re.compile(r"[^\s,]+") #one or more char not whitespace comma
}
rx_dict_SELECT = {
    'DATA'      : re.compile(r"\.DATA\d{1,2}\(([^\)]*)\)"),
    'CONTROL'   : re.compile(r"\.CONTROL\d{1,2}\(([^\)]*)\)"),
    'Z'         : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_comp = {
    'A'             : re.compile(r"\.A\(([^\)]*)\)"),
    'B'             : re.compile(r"\.B\(([^\)]*)\)"),
    'QUOTIENT'      : re.compile(r"\.QUOTIENT\(([^\)]*)\)")
}
rx_dict_SUB_ADD_MULT = {
    'A'     : re.compile(r"\.A\(([^\)]*)\)"),
    'B'     : re.compile(r"\.B\(([^\)]*)\)"),
    'Z'     : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_MUX = {
    'D'     : re.compile(r"\.D\d{1,2}\(([^\)]*)\)"), # SEEMS TO BE SOME THAT HAS UP TO D31 AS D INPUTS. HOW DO i HANDLE THese VARYING THINgS
                                                    # ALSO SOME ONLY GOING TO D3 BUT ATTACHING 5 BIT TO EACH D data width of d varies, number of D inputs varies
    'S'     : re.compile(r"\.S\d{1,2}\(([^\)]*)\)"), # make arrays for the mux
    'Z'     : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_BUF = {
    'A'     : re.compile(r"\.A\(([^\)]*)\)"),
    'Z'     : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_NOT = {
    'A'     : re.compile(r"\.A\(([^\)]*)\)"),
    'Z'     : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_AND2  = {
    'A'     : re.compile(r"\.A\(([^\)]*)\)"),
    'B'     : re.compile(r"\.B\(([^\)]*)\)"),
    'Z'     : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_OR2 = {
    'A'     : re.compile(r"\.A\(([^\)]*)\)"),
    'B'     : re.compile(r"\.B\(([^\)]*)\)"),
    'Z'     : re.compile(r"\.Z\(([^\)]*)\)")
}
rx_dict_reg = {
    'clear'           : re.compile(r"\.clear\(([^\)]*)\)"),
    'preset'          : re.compile(r"\.preset\(([^\)]*)\)"),
    'next_state'      : re.compile(r"\.next_state\(([^\)]*)\)"),
    'clocked_on'      : re.compile(r"\.clocked_on\(([^\)]*)\)"),
    'data_in'         : re.compile(r"\.data_in\(([^\)]*)\)"),
    'enable'          : re.compile(r"\.enable\(([^\)]*)\)"),
    'Q'               : re.compile(r"\.Q\(([^\)]*)\)"),
    'QN'              : re.compile(r"\.QN\(([^\)]*)\)"),
    'synch_clear'     : re.compile(r"\.synch_clear\(([^\)]*)\)"),
    'synch_preset'    : re.compile(r"\.synch_preset\(([^\)]*)\)"),
    'synch_toggle'    : re.compile(r"\.synch_toggle\(([^\)]*)\)"),
    'synch_enable'    : re.compile(r"\.synch_enable\(([^\)]*)\)")
}

def features(module_name):

    print(f"in module: {module_name}")
    global top_level_parents

    #header = ['reg_name','1','2']
    #header = ['reg_name']
    #first_row = feature_in_one(top_level_parents[0])

    #for group_name, feature_dict in first_row.items():
    #    for feature_name in feature_dict.keys():
    #        header.append(f"{group_name}_{feature_name}")
    
    with open(reg_features_temp_file, 'a', newline = '') as f:
        writer = csv.writer(f)
        #writer.writerow(header)

        for s in top_level_parents:
            if s.represented_object_handle.module_belong == module_name:

                tree_feature = feature_in_one(s)
                #feature1,feature2 = feature_in_all(s)
                row_temp = [s.represented_object_handle.longname]
                #row_temp = [s.represented_object_handle.longname,feature1,feature2]
            
                for group_name, feature_dict in tree_feature.items():
                    for feature_name in feature_dict.keys():
                        row_temp.append(feature_dict[feature_name])

                writer.writerow(row_temp)

                #feature_clear(s) # clear features of all structures below reg s every time one s is processed

    #for s in top_level_parents:# clear features of all structures below the module every time one module is processed
    #    if s.represented_object_handle.module_belong == module_name:
    #        feature_clear(s)

def feature_clear(s):

    s.leaf_type.clear()
    s.width.clear()
    s.depth.clear()
    s.leaf_type_intree.clear()
    s.width_intree.clear()
    s.depth_intree.clear()
    s.feature_detected = False

    s_children = children_replace(s.children)
    for child in s_children:
        feature_clear(child)
     

def feature_extract(leaf_type, width, depth):

    width_max = 0
    width_ave = 0
    width_std = 0
    width_max_level = 0

    depth_max = 0
    depth_ave = 0
    depth_std = 0

    node_num = 0
    node_leaf_num = 0
    node_children_num_max = 0
    node_children_num_ave = 0
    
    gtech_node_num = {
        "gtech_or2"   : 0,
        "gtech_and2"  : 0,
        "gtech_xor2"  : 0,
        "gtech_not"   : 0,
        "gtech_buf"   : 0,
        "select_op"   : 0,
        "mux_op"      : 0,
        "comp_op"     : 0,
        "sub_op"      : 0,
        "add_op"      : 0,
        "mult_op"     : 0,
        "div_op"      : 0,
        "shift_op"    : 0
    }

    # no use
    gtech_node_children_num_max = {
        "gtech_or2"   : 0,
        "gtech_and2"  : 0,
        "gtech_xor2"  : 0,
        "gtech_not"   : 0,
        "gtech_buf"   : 0,
        "select_op"   : 0,
        "mux_op"      : 0,
        "comp_op"     : 0,
        "sub_op"      : 0,
        "add_op"      : 0,
        "mult_op"     : 0,
        "div_op"      : 0,
        "shift_op"    : 0
    }

    gtech_node_children_num_ave = {
        "gtech_or2"   : 0,
        "gtech_and2"  : 0,
        "gtech_xor2"  : 0,
        "gtech_not"   : 0,
        "gtech_buf"   : 0,
        "select_op"   : 0,
        "mux_op"      : 0,
        "comp_op"     : 0,
        "sub_op"      : 0,
        "add_op"      : 0,
        "mult_op"     : 0,
        "div_op"      : 0,
        "shift_op"    : 0
    }

    gtech_node_ave_level = {
        "gtech_or2"   : 0,
        "gtech_and2"  : 0,
        "gtech_xor2"  : 0,
        "gtech_not"   : 0,
        "gtech_buf"   : 0,
        "select_op"   : 0,
        "mux_op"      : 0,
        "comp_op"     : 0,
        "sub_op"      : 0,
        "add_op"      : 0,
        "mult_op"     : 0,
        "div_op"      : 0,
        "shift_op"    : 0
    }

    tree_feature = {
        "width_max" : 0,
        "width_ave" : 0,
        "width_std" : 0,
        "width_max_level" : 0,
        "depth_max" : 0,
        "depth_ave" : 0,
        "depth_std" : 0,
        "node_num" : 0,
        "node_leaf_num" : 0,
        #"node_children_num_max" : 0,
        "node_children_num_ave" : 0
    }

    width_all = reduce(np.add,width.values())
    width_max = max(width_all)
    width_ave = np.sum(width_all)/width_all.size
    width_std = np.std(width_all)
    width_max_level = np.argmax(width_all)
    
    depth_all = reduce(np.add,depth.values())
    depth_max = max(depth_all)
    #if depth_max + 1 != width_all.size:
    #    print(depth_max)
    #    print(width_all.size)
    #    print("hey, is there something wrong?")
    depth_ave = sum(depth_all)/depth_all.size
    depth_std = np.std(depth_all)

    gtech_node_children_num_all = []
    for key in type_dict:

        gtech_node_num[key] = np.sum(width[key])

        gtech_node_children_num_key = leaf_type[key]
        gtech_node_children_num_all.append(np.sum(gtech_node_children_num_key))
        if gtech_node_num[key] == 0:
            gtech_node_children_num_ave[key] = 0
        else:
            #gtech_node_children_num_max[key] = max(gtech_node_children_num_key)
            gtech_node_children_num_ave[key] = np.sum(gtech_node_children_num_key)/gtech_node_num[key]

        gtech_node_ave_level[key] =  np.sum(width[key] * np.arange(width[key].size)) / width[key].size

    node_num = np.sum(list(gtech_node_num.values()))
    node_leaf_num = depth_all.size
    #node_children_num_max = max(list(gtech_node_children_num_max.values()))
    if node_num == 0:
        node_children_num_ave = 0
    else:
        node_children_num_ave = sum(gtech_node_children_num_all)/node_num

    tree_feature["width_max"] = width_max
    tree_feature["width_ave"] = width_ave
    tree_feature["width_std"] = width_std
    tree_feature["width_max_level"] = width_max_level
    tree_feature["depth_max"] = depth_max
    tree_feature["depth_ave"] = depth_ave
    tree_feature["depth_std"] = depth_std
    tree_feature["node_num"] = node_num
    tree_feature["node_leaf_num"] = node_leaf_num
    #tree_feature["node_children_num_max"] = node_children_num_max
    tree_feature["node_children_num_ave"] = node_children_num_ave

    # check:
    if tree_feature["node_num"] < tree_feature["width_ave"] * tree_feature["depth_ave"]:
        tree_feature["node_num"] = tree_feature["width_ave"] * tree_feature["depth_ave"]
        for key in type_dict:
            gtech_node_num[key] = tree_feature["node_num"] / 13
            gtech_node_num[key] = 1 / 13
            gtech_node_ave_level[key] = tree_feature["depth_ave"] / 2 


    #return tree_feature, gtech_node_num, gtech_node_children_num_max, gtech_node_children_num_ave, gtech_node_ave_level
    return tree_feature, gtech_node_num, gtech_node_children_num_ave, gtech_node_ave_level

    # tai ma fan le , fang qi
    gtech_node_num_grouped = {
        "gtech_or2"   : gtech_node_num["gtech_or2"],
        "gtech_and2"  : gtech_node_num["gtech_and2"],
        "gtech_xor2"  : gtech_node_num["gtech_xor2"],
        "gtech_not"   : gtech_node_num["gtech_not"],
        "gtech_buf"   : gtech_node_num["gtech_buf"],
        "select_mux"  : gtech_node_num["select_op"] + gtech_node_num["mux_op"], # slect_op and mux_op
        "comp_sub"    : gtech_node_num["comp_op"] + gtech_node_num["sub_op"], # comp_op and sub_op
        "add_op"      : gtech_node_num["add_op"],
        "mult_shift"  : gtech_node_num["mult_op"] + gtech_node_num["shift_op"], # mult_op and shift_op
        "div_op"      : gtech_node_num["div_op"]
    }

    gtech_node_children_num_ave_grouped = {
        "gtech_or2"   : gtech_node_children_num_ave["gtech_or2"],
        "gtech_and2"  : gtech_node_children_num_ave["gtech_and2"],
        "gtech_xor2"  : gtech_node_children_num_ave["gtech_xor2"],
        "gtech_not"   : gtech_node_children_num_ave["gtech_not"],
        "gtech_buf"   : gtech_node_children_num_ave["gtech_buf"],
        "select_mux"  : 0, # slect_op and mux_op
        "comp_sub"    : 0, # comp_op and sub_op
        "add_op"      : gtech_node_children_num_ave["add_op"],
        "mult_shift"  : 0, # mult_op and shift_op
        "div_op"      : gtech_node_children_num_ave["div_op"]
    }

    gtech_node_ave_level_grouped = {
        "gtech_or2"   : gtech_node_ave_level["gtech_or2"],
        "gtech_and2"  : gtech_node_ave_level["gtech_and2"],
        "gtech_xor2"  : gtech_node_ave_level["gtech_xor2"],
        "gtech_not"   : gtech_node_ave_level["gtech_not"],
        "gtech_buf"   : gtech_node_ave_level["gtech_buf"],
        "select_mux"  : 0, # slect_op and mux_op
        "comp_sub"    : 0, # comp_op and sub_op
        "add_op"      : gtech_node_ave_level["add_op"],
        "mult_shift"  : 0, # mult_op and shift_op
        "div_op"      : gtech_node_ave_level["div_op"]
    }


    if gtech_node_num_grouped["select_mux"] == 0:
        gtech_node_children_num_ave_grouped["select_mux"] = 0
        gtech_node_ave_level_grouped["select_mux"] = 0
    else:
        gtech_node_children_num_ave_grouped["select_mux"] = (gtech_node_children_num_ave["select_op"] * gtech_node_num["select_op"] + \
                        gtech_node_children_num_ave["mux_op"] * gtech_node_num["mux_op"]) / (gtech_node_num_grouped["select_mux"])

        gtech_node_ave_level_grouped["select_mux"] = (gtech_node_ave_level["select_op"] * gtech_node_num["select_op"] + \
                        gtech_node_ave_level["mux_op"] * gtech_node_num["mux_op"]) / (gtech_node_num_grouped["select_mux"])

    if gtech_node_num_grouped["comp_sub"] == 0:
        gtech_node_children_num_ave_grouped["comp_sub"] = 0
        gtech_node_ave_level_grouped["comp_sub"] = 0
    else:
        gtech_node_children_num_ave_grouped["comp_sub"] = (gtech_node_children_num_ave["comp_op"] * gtech_node_num["comp_op"] + \
                        gtech_node_children_num_ave["sub_op"] * gtech_node_num["sub_op"]) / (gtech_node_num_grouped["comp_sub"])

        gtech_node_ave_level_grouped["comp_sub"] = (gtech_node_ave_level["comp_op"] * gtech_node_num["comp_op"] + \
                        gtech_node_ave_level["sub_op"] * gtech_node_num["sub_op"]) / (gtech_node_num_grouped["comp_sub"])

    if gtech_node_num_grouped["mult_shift"] == 0:
        gtech_node_children_num_ave_grouped["mult_shift"] = 0
        gtech_node_ave_level_grouped["mult_shift"] = 0
    else:
        gtech_node_children_num_ave_grouped["mult_shift"] = (gtech_node_children_num_ave["mult_op"] * gtech_node_num["mult_op"] + \
                        gtech_node_children_num_ave["shift_op"] * gtech_node_num["shift_op"]) / (gtech_node_num_grouped["mult_shift"])

        gtech_node_ave_level_grouped["mult_shift"] = (gtech_node_ave_level["mult_op"] * gtech_node_num["mult_op"] + \
                        gtech_node_ave_level["shift_op"] * gtech_node_num["shift_op"]) / (gtech_node_num_grouped["mult_shift"])

    #return tree_feature, gtech_node_num_grouped, gtech_node_children_num_ave_grouped, gtech_node_ave_level_grouped


def reg_feature_extract(reg_width, reg_depth, reg_children_num):

    reg_tree_feature = {
        "width_max" : 0,
        "width_ave" : 0,
        "width_std" : 0,
        "width_max_level" : 0,
        "depth_max" : 0,
        "depth_ave" : 0,
        "depth_std" : 0,
        "node_num" : 0,
        "node_leaf_num" : 0,
        "node_children_num_ave" : 0
    }
    #print(reg_depth)

    width_max = 0
    width_ave = 0
    width_std = 0
    width_max_level = 0

    depth_max = 0
    depth_ave = 0
    depth_std = 0

    node_num = 0
    node_leaf_num = 0
    node_children_num_ave = 0

    width_max = max(reg_width)
    width_ave = np.sum(reg_width)/reg_width.size
    width_std = np.std(reg_width)
    width_max_level = np.argmax(reg_width)

    depth_max = max(reg_depth)
    depth_ave = sum(reg_depth)/reg_depth.size
    depth_std = np.std(reg_depth)

    node_num = np.sum(reg_width)
    node_leaf_num = reg_depth.size
    node_children_num_ave = reg_children_num / node_num

    reg_tree_feature["width_max"] = width_max
    reg_tree_feature["width_ave"] = width_ave
    reg_tree_feature["width_std"] = width_std
    reg_tree_feature["width_max_level"] = width_max_level
    reg_tree_feature["depth_max"] = depth_max
    reg_tree_feature["depth_ave"] = depth_ave
    reg_tree_feature["depth_std"] = depth_std
    reg_tree_feature["node_num"] = node_num
    reg_tree_feature["node_leaf_num"] = node_leaf_num
    reg_tree_feature["node_children_num_ave"] = node_children_num_ave

    return reg_tree_feature


def feature_in_one(s):

    print(f"find feature of {s.represented_object_handle.longname}")

    leaf_type, width, depth, leaf_type_intree, width_intree, depth_intree = find_feature_in_one(s)

    #reg_width, reg_depth, reg_children_num, reg_width_intree, reg_depth_intree, reg_children_num_intree  = find_feature_in_all(s)  
    #print(reg_depth)

    #tree_feature, gtech_node_num, gtech_node_children_num_max, gtech_node_children_num_ave, gtech_node_ave_level =\
    tree_feature, gtech_node_num, gtech_node_children_num_ave, gtech_node_ave_level =\
     feature_extract(leaf_type, width, depth)

    #tree_feature_intree, gtech_node_num_intree, gtech_node_children_num_max_intree, gtech_node_children_num_ave_intree, gtech_node_ave_level_intree =\
    tree_feature_intree, gtech_node_num_intree, gtech_node_children_num_ave_intree, gtech_node_ave_level_intree =\
     feature_extract(leaf_type_intree, width_intree, depth_intree)

    #reg_tree_feature = reg_feature_extract(reg_width, reg_depth, reg_children_num)
    #reg_tree_feature_intree = reg_feature_extract(reg_width_intree, reg_depth_intree, reg_children_num_intree)
    
    #$return tree_feature, gtech_node_num, gtech_node_children_num_ave, gtech_node_ave_level,\
    # tree_feature_intree, gtech_node_num_intree, gtech_node_children_num_ave_intree, gtech_node_ave_level_intree

    gtech_real_num = copy.deepcopy(s.gtech_real_num)
    #print(gtech_real_num)
    gtech_real_depth = copy.deepcopy(s.gtech_real_depth)
    for key in type_dict:
        if gtech_real_num[key] == 0:
            gtech_real_depth[key] = 0.0
        else:
            gtech_real_depth[key] = gtech_real_depth[key] / gtech_real_num[key]

    return OrderedDict({
        "tree_feature" : tree_feature,
        "tree_feature_intree" : tree_feature_intree,
        #"reg_tree_feature" : reg_tree_feature,
        #"reg_tree_feature_intree" : reg_tree_feature_intree,
        "gtech_real_num" : gtech_real_num,
        "gtech_real_depth" : gtech_real_depth,
        "gtech_node_num" : gtech_node_num,
        "gtech_node_num_intree" : gtech_node_num_intree,  
        "gtech_node_children_num_ave" : gtech_node_children_num_ave, 
        "gtech_node_children_num_ave_intree" : gtech_node_children_num_ave_intree, 
        "gtech_node_ave_level" : gtech_node_ave_level,
        "gtech_node_ave_level_intree" : gtech_node_ave_level_intree
    })


def find_feature_in_one(s):

    global level
    level = level + 1

    if s.feature_detected == True:

        leaf_type = copy.deepcopy(s.leaf_type)
        width = copy.deepcopy(s.width)
        depth = copy.deepcopy(s.depth)
        leaf_type_intree = copy.deepcopy(s.leaf_type_intree)
        width_intree = copy.deepcopy(s.width_intree)
        depth_intree = copy.deepcopy(s.depth_intree)

        sub_intree(s)

        return leaf_type, width, depth, leaf_type_intree, width_intree, depth_intree

    leaf_type_temp = {
        "gtech_or2"   : 0,
        "gtech_and2"  : 0,
        "gtech_xor2"  : 0,
        "gtech_not"   : 0,
        "gtech_buf"   : 0,
        "select_op"   : 0,
        "mux_op"      : 0,
        "comp_op"     : 0,
        "sub_op"      : 0,
        "add_op"      : 0,
        "mult_op"     : 0,
        "div_op"      : 0,
        "shift_op"    : 0
    }

    leaf_type_intree_temp = {
        "gtech_or2"   : 0,
        "gtech_and2"  : 0,
        "gtech_xor2"  : 0,
        "gtech_not"   : 0,
        "gtech_buf"   : 0,
        "select_op"   : 0,
        "mux_op"      : 0,
        "comp_op"     : 0,
        "sub_op"      : 0,
        "add_op"      : 0,
        "mult_op"     : 0,
        "div_op"      : 0,
        "shift_op"    : 0
    }

    leaf_type = {
        "gtech_or2"   : np.array([]),
        "gtech_and2"  : np.array([]),
        "gtech_xor2"  : np.array([]),
        "gtech_not"   : np.array([]),
        "gtech_buf"   : np.array([]),
        "select_op"   : np.array([]),
        "mux_op"      : np.array([]),
        "comp_op"     : np.array([]),
        "sub_op"      : np.array([]),
        "add_op"      : np.array([]),
        "mult_op"     : np.array([]),
        "div_op"      : np.array([]),
        "shift_op"    : np.array([])
    }

    width = {
        "gtech_or2"   : np.array([]),
        "gtech_and2"  : np.array([]),
        "gtech_xor2"  : np.array([]),
        "gtech_not"   : np.array([]),
        "gtech_buf"   : np.array([]),
        "select_op"   : np.array([]),
        "mux_op"      : np.array([]),
        "comp_op"     : np.array([]),
        "sub_op"      : np.array([]),
        "add_op"      : np.array([]),
        "mult_op"     : np.array([]),
        "div_op"      : np.array([]),
        "shift_op"    : np.array([])
    }

    depth = {
        "gtech_or2"   : np.array([]),
        "gtech_and2"  : np.array([]),
        "gtech_xor2"  : np.array([]),
        "gtech_not"   : np.array([]),
        "gtech_buf"   : np.array([]),
        "select_op"   : np.array([]),
        "mux_op"      : np.array([]),
        "comp_op"     : np.array([]),
        "sub_op"      : np.array([]),
        "add_op"      : np.array([]),
        "mult_op"     : np.array([]),
        "div_op"      : np.array([]),
        "shift_op"    : np.array([]),
    }

    leaf_type_intree = {
        "gtech_or2"   : np.array([]),
        "gtech_and2"  : np.array([]),
        "gtech_xor2"  : np.array([]),
        "gtech_not"   : np.array([]),
        "gtech_buf"   : np.array([]),
        "select_op"   : np.array([]),
        "mux_op"      : np.array([]),
        "comp_op"     : np.array([]),
        "sub_op"      : np.array([]),
        "add_op"      : np.array([]),
        "mult_op"     : np.array([]),
        "div_op"      : np.array([]),
        "shift_op"    : np.array([])
    }

    width_intree = {
        "gtech_or2"   : np.array([]),
        "gtech_and2"  : np.array([]),
        "gtech_xor2"  : np.array([]),
        "gtech_not"   : np.array([]),
        "gtech_buf"   : np.array([]),
        "select_op"   : np.array([]),
        "mux_op"      : np.array([]),
        "comp_op"     : np.array([]),
        "sub_op"      : np.array([]),
        "add_op"      : np.array([]),
        "mult_op"     : np.array([]),
        "div_op"      : np.array([]),
        "shift_op"    : np.array([])
    }

    depth_intree = {
        "gtech_or2"   : np.array([]),
        "gtech_and2"  : np.array([]),
        "gtech_xor2"  : np.array([]),
        "gtech_not"   : np.array([]),
        "gtech_buf"   : np.array([]),
        "select_op"   : np.array([]),
        "mux_op"      : np.array([]),
        "comp_op"     : np.array([]),
        "sub_op"      : np.array([]),
        "add_op"      : np.array([]),
        "mult_op"     : np.array([]),
        "div_op"      : np.array([]),
        "shift_op"    : np.array([])
    }

    s_children = children_replace(s.children)

    if s_children != []:
        for child in s_children:

            leaf_type_temp[child.represented_object_handle.id] += 1
            leaf_type_intree_temp[child.represented_object_handle.id] += 1/child.intree

            leaf_type_results, width_results, depth_results, leaf_type_intree_results, width_intree_results, depth_intree_results = find_feature_in_one(child)

            leaf_type_result = copy.deepcopy(leaf_type_results)
            width_result = copy.deepcopy(width_results)
            depth_result = copy.deepcopy(depth_results)
            leaf_type_intree_result = copy.deepcopy(leaf_type_intree_results)
            width_intree_result = copy.deepcopy(width_intree_results)
            depth_intree_result = copy.deepcopy(depth_intree_results)

            for key in type_dict:

                if leaf_type[key].size == 0:
                    leaf_type[key] = leaf_type_result[key]
                else:
                    leaf_type[key] = leaf_type_result[key] + leaf_type[key]

                if leaf_type_intree[key].size == 0:
                    leaf_type_intree[key] = leaf_type_intree_result[key]
                else:
                    leaf_type_intree[key] = leaf_type_intree_result[key] + leaf_type_intree[key]


                if depth[key].size == 0:
                    depth[key] = depth_result[key]
                else:
                    depth[key] = np.concatenate((depth[key],depth_result[key]),axis = 0)

                if depth_intree[key].size == 0:
                    depth_intree[key] = depth_intree_result[key]
                else:
                    depth_intree[key] = np.concatenate((depth_intree[key],depth_intree_result[key]),axis = 0)


                if width[key].size == 0:
                    width[key] = width_result[key]
                else:
                    if width[key].size < width_result[key].size:
                        width[key] = np.pad(width[key],(0,width_result[key].size - width[key].size),mode = 'constant')
                    elif width[key].size > width_result[key].size:
                        width_result[key] = np.pad(width_result[key],(0,width[key].size - width_result[key].size),mode = 'constant')
                    width[key] = width[key] + width_result[key]

                if width_intree[key].size == 0:
                    width_intree[key] = width_intree_result[key]
                else:
                    if width_intree[key].size < width_intree_result[key].size:
                        width_intree[key] = np.pad(width_intree[key],(0,width_intree_result[key].size - width_intree[key].size),mode = 'constant')
                    elif width_intree[key].size > width_intree_result[key].size:
                        width_intree_result[key] = np.pad(width_intree_result[key],(0,width_intree[key].size - width_intree_result[key].size),mode = 'constant')
                    width_intree[key] = width_intree[key] + width_intree_result[key]


        leaf_type_this = np.array([list(leaf_type_temp.values())])
        leaf_type_intree_this = np.array([list(leaf_type_intree_temp.values())])


        for key in type_dict:
            if key == s.represented_object_handle.id:
                width[key] = np.concatenate((np.array([1]),width[key]),axis = 0)
                width_intree[key] = np.concatenate((np.array([1/s.intree]),width_intree[key]),axis = 0)
            else:
                width[key] = np.concatenate((np.array([0]),width[key]),axis = 0)
                width_intree[key] = np.concatenate((np.array([0]),width_intree[key]),axis = 0)

        if s.represented_object_handle.id in type_dict:
            depth[s.represented_object_handle.id] += 1
            depth_intree[s.represented_object_handle.id] += 1/s.intree
            leaf_type[s.represented_object_handle.id] = leaf_type[s.represented_object_handle.id] + leaf_type_this
            leaf_type_intree[s.represented_object_handle.id] = leaf_type_intree[s.represented_object_handle.id] + leaf_type_intree_this
        
    else:
        for key in type_dict:
            leaf_type[key] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
            leaf_type_intree[key] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0],dtype = np.float64)
            width[key] = np.array([0])
            width_intree[key] = np.array([0],dtype = np.float64)
            depth[key] = np.array([0])
            depth_intree[key] = np.array([0],dtype = np.float64)

        if s.represented_object_handle.id in type_dict:
            depth[s.represented_object_handle.id] = np.array([1])
            depth_intree[s.represented_object_handle.id] = np.array([1/s.intree],dtype = np.float64)
            width[s.represented_object_handle.id] = np.array([1])
            width_intree[s.represented_object_handle.id] = np.array([1/s.intree],dtype = np.float64)

    #print(len(s.parents))

    s.intree_count = s.intree_count - 1
    
    if s.intree_count > 0:
    #if s.intree_count > 0 and level % 50 == 0:

        s.leaf_type = leaf_type
        s.width = width
        s.depth = depth
        s.leaf_type_intree = leaf_type_intree
        s.width_intree = width_intree
        s.depth_intree = depth_intree
        s.feature_detected = True
        
    level = level - 1

    return leaf_type, width, depth, leaf_type_intree, width_intree, depth_intree

# run toooooooo slow ! the tree is toooooooooooooooooooooooooooo large !
def find_feature_in_all(s):

    #print(f"    find feature of {s.represented_object_handle.longname}")

    global reg_stack
    reg_stack.append(s)

    global level
    level = level + 1
    print(f"level = {level}")

    reg_children_num_temp = 0

    reg_width = np.array([])
    reg_depth = np.array([])
    reg_children_num = 0 # sum of all the children's children num

    reg_width_intree = np.array([])
    reg_depth_intree = np.array([])
    reg_children_num_intree = 0 # sum of all the children's children num

    if s.reg_feature_detected == True:

        print("1")
        reg_width = copy.deepcopy(s.reg_width)
        reg_depth = copy.deepcopy(s.reg_depth)
        reg_children_num = copy.deepcopy(s.reg_children_num)
        reg_width_intree = copy.deepcopy(s.reg_width_intree)
        reg_depth_intree = copy.deepcopy(s.reg_depth_intree)
        reg_children_num_intree = copy.deepcopy(s.reg_children_num_intree)

    #    print("2")
    #    reg_stack.pop()
    #    print("3")

        return  reg_width, reg_depth, reg_children_num, reg_width_intree, reg_depth_intree, reg_children_num_intree


    s_reg_children = reg_children_replace(s.reg_children)


    reg_children_num = len(s_reg_children)
    reg_children_num_intree = len(s_reg_children) / s.intree

    #print("cccccccccccccccccccccccc")
    #print(s.represented_object_handle.longname)
    #print(len(s_reg_children))

    if s_reg_children != [] and level < 10:

        for child in s_reg_children:

            #print(child.represented_object_handle.longname)

            reg_width_results, reg_depth_results, reg_children_num_results, reg_width_intree_results, reg_depth_intree_results, reg_children_num_intree_results = find_feature_in_all(child)


            #print("4")
            reg_width_result = copy.deepcopy(reg_width_results)
            reg_depth_result = copy.deepcopy(reg_depth_results)
            reg_children_num_result = copy.deepcopy(reg_children_num_results)
            reg_width_intree_result = copy.deepcopy(reg_width_intree_results)
            reg_depth_intree_result = copy.deepcopy(reg_depth_intree_results)
            reg_children_num_intree_result = copy.deepcopy(reg_children_num_intree_results)

            reg_children_num = reg_children_num + reg_children_num_result
            reg_children_num_intree = reg_children_num_intree + reg_children_num_intree_result

            #print("5")
            if reg_width.size == 0:
                reg_width = reg_width_result
            else:
                if reg_width.size < reg_width_result.size:
                    reg_width = np.pad(reg_width,(0,reg_width_result.size - reg_width.size),mode = 'constant')
                elif reg_width.size > reg_width_result.size:
                    reg_width_result = np.pad(reg_width_result,(0,reg_width.size - reg_width_result.size),mode = 'constant')
                reg_width = reg_width + reg_width_result

            if reg_width_intree.size == 0:
                reg_width_intree = reg_width_intree_result
            else:
                if reg_width_intree.size < reg_width_intree_result.size:
                    reg_width_intree = np.pad(reg_width_intree,(0,reg_width_intree_result.size - reg_width_intree.size),mode = 'constant')
                elif reg_width_intree.size > reg_width_intree_result.size:
                    reg_width_intree_result = np.pad(reg_width_intree_result,(0,reg_width_intree.size - reg_width_intree_result.size),mode = 'constant')
                reg_width_intree = reg_width_intree + reg_width_intree_result

            #print("6")
            if reg_depth.size == 0:
                reg_depth = reg_depth_result
            else:
                reg_depth = np.concatenate((reg_depth,reg_depth_result),axis = 0)

            if reg_depth_intree.size == 0:
                reg_depth_intree = reg_depth_intree_result
            else:
                reg_depth_intree = np.concatenate((reg_depth_intree,reg_depth_intree_result),axis = 0)
            #print("7")

        reg_width = np.concatenate((np.array([1]),reg_width),axis = 0)
        reg_width_intree = np.concatenate((np.array([1/s.intree]),reg_width_intree),axis = 0)
        reg_depth += 1
        reg_depth_intree += 1/s.intree

    else:
        reg_width = np.array([1])
        reg_depth = np.array([1])
        reg_children_num = 0
        reg_width_intree = np.array([1/s.intree],dtype = np.float64)
        reg_depth_intree = np.array([1/s.intree],dtype = np.float64)
        reg_children_num_intree = 0.0

    s.reg_width = reg_width
    s.reg_depth = reg_depth
    s.reg_children_num = reg_children_num
    s.reg_width_intree = reg_width_intree
    s.reg_depth_intree = reg_depth_intree
    s.reg_children_num_intree = reg_children_num_intree

    s.reg_feature_detected = True

    level = level - 1

    print(f"levelend = {level}")
    reg_stack.pop()

    return reg_width, reg_depth, reg_children_num, reg_width_intree, reg_depth_intree, reg_children_num_intree 


def children_replace(s_children):

    s_children_temp = []

    for s in s_children:

        if s.represented_object_handle.id in type_dict:
            s_children_temp.append(s)

        elif s.represented_object_handle.id == 'input' or s.represented_object_handle.id == 'unknown':
            s_children_temp.extend(children_replace(s.children))

    return s_children_temp


def reg_children_replace(s_reg_children):

    global reg_stack
    s_reg_children_temp = []

    for s in s_reg_children:

        if s not in reg_stack and s.represented_object_handle.id != 'input':
            s_reg_children_temp.append(s)

        elif s.represented_object_handle.id == 'input' or s.represented_object_handle.id == 'unknown':
            s_reg_children_temp.extend(reg_children_replace(s.reg_children))

    return s_reg_children_temp


type_dict = [
    "gtech_or2",
    "gtech_and2",
    "gtech_xor2",
    "gtech_not",
    "gtech_buf",
    "select_op",
    "mux_op",
    "comp_op",
    "sub_op",
    "add_op",
    "mult_op",
    "div_op",
    "shift_op"
]









run_parse_elab(filename,top_module_name)



