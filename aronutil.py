# -*- coding: utf-8 -*-
"""
Created on Wed May 23 08:28:46 2018

@author: tordaronsen
"""
import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from scipy.constants import g
import matplotlib.pyplot as plt
import zipfile

blocks = [
    'STRESS_LINE_LIST:Local_section_forces.Max_axial_force_N {', 
    'STRESS_LINE_LIST:Local_section_forces.Max_axial_force_N_INDEX {',
    'STRESS_LINE_LIST:Global_section_forces.Max_force_Z_N {',
    'STRESS_LINE_LIST:Global_section_forces.Max_force_Z_N_INDEX {',
    'STRESS_LINE_LIST:Nominal_stress_range.Right_web_MPa {',
    'STRESS_LINE_LIST:Nominal_stress_range.Right_web_MPa_INDEX {',
    'STRESS_LINE_LIST:Convergence_norm {',
    'STRESS_LINE_LIST:Convergence_norm_INDEX {'
] 

block_map =  {
    blocks[0]: 'Forces',
    blocks[1]: 'Force_indices',
    blocks[2]: 'Z_forces',
    blocks[3]: 'Z_forces_indices',
    blocks[4]: 'Right_web',
    blocks[5]: 'Right_web_indices',
    blocks[5]: 'Right_web_indices',
    blocks[6]: 'Conv_norm',
    blocks[7]: 'Conv_norm_indices'
}

def model(path, is_accident, is_nice=False, to_clipboard=False):
    '''is_nice is true when names are clever (or nice!). 
    When is_nice=True components are categorized.
    When to_clipboard is true MBL, Materialcoeff and Materials 
    are written to clipboard.'''
    
    if path[-4:] == ".avz":        
        with zipfile.ZipFile(file=path) as zfile:
            with zfile.open('model.xml') as file:
                root = et.XML(file.read().decode("Latin-1"))
    elif path[-4:] == ".xml":
        tree = et.parse("Testmiljø/Horsvaagen_90m_.xml")
        root = tree.getroot()
    else:
        print("Input file must be either .avz or .xml.")
        return None
    
    model = {
        "Lastgrense [tonn]": [],
        "Edit ID": [], 
        "Materialkoeffisient": [], 
        "MBL [tonn]": [],
        "Navn": [],
        "Komponent": [],
        "Materiale": [],
        "ID": []
    }
    
    for comp in root.iter("component"):        
        mcoeff      = float(comp.attrib["materialcoeff"])
        mbl         = float(comp.attrib["breakingload"])/(g*1000)
        name        = comp.attrib["name"]
        name_list   = name.split(":")
        model["Komponent"].append(name_list[0].strip())
        model["Materiale"].append(name_list[-1].strip())
        model["Navn"].append(name)
        model["Materialkoeffisient"].append(float(comp.attrib["materialcoeff"]))
        model["MBL [tonn]"].append(mbl)
        model["ID"].append(int(comp.attrib["id"]))
        model["Edit ID"].append(int(comp.attrib["number"]))
        if is_accident:
            model["Lastgrense [tonn]"].append(mbl/(mcoeff/1.5))
        else:
            model["Lastgrense [tonn]"].append(mbl/(mcoeff*1.15))
    
    df_model = pd.DataFrame(data = model)
    df_model.set_index("ID", inplace=True)
    
    if is_nice:
        df_model[['Komponent', 'Gruppe']] = df_model.Komponent.str.split('_', expand=True)
        df_model.loc[:,'Gruppe'] = df_model.Gruppe.astype('category')
    
    return df_model

def collect_avz_data(avz_path, blocks):
    '''Parse data from file to a dictionary.'''
    with zipfile.ZipFile(avz_path) as zfile:
        with zfile.open('model.avs') as file:
            is_inside = False
            content = []
            data_dicts = {}
            data_key = '' # Empty byte string
            for line in file:
                nice_line = line.decode('Latin-1').strip()
                for block_name in blocks:
                    if block_name in nice_line:
                        is_inside = True
                        data_key = block_map[block_name]
                        if data_key not in data_dicts:
                            data_dicts[data_key] = {}

                if '}' in nice_line and is_inside:
                    is_inside = False
                    data_dicts[data_key][content[1]] = [np.float64(text.split()[-1]) for text in content[2:]]
                    content.clear()

                if is_inside:
                    content.append(nice_line)
    return data_dicts

def avz_result(data_dicts, return_df_data=False):
    '''Make DataFrame from data Dictionary. 
    If return_df_data is true, df_data DataFrame is also returned.'''
    df_data = pd.DataFrame(data_dicts)
    df_data.reset_index(inplace=True)
    df_data.set_index(df_data['index'].str.split(expand=True)[1].astype(np.int64),
                         inplace=True)
    df_data.sort_index(inplace=True)
    # The following approach may not be correct for membrane or beam.
    # The stress blocks has two columns which are (seemingly) equal for truss.
    df_data['Forces_argmax'] = df_data.Forces.apply(np.argmax).astype(np.int64)
    df_data['Z_forces_argmax'] = df_data.Z_forces.apply(np.argmax).astype(np.int64)
    df_data['Z_forces_argmin'] = df_data.Z_forces.apply(np.argmin).astype(np.int64)
    df_data['Right_web_argmax'] = df_data.Right_web.apply(np.argmax).astype(np.int64)
    df_data['Conv_norm_argmax'] = df_data.Conv_norm.apply(np.argmax).astype(np.int64)
    
    df_max = pd.DataFrame(index=df_data.index)
    df_max.index.name = 'ID'
    df_max['Last [N]'] = df_data.apply(lambda row: row['Forces'][row['Forces_argmax']], axis=1)
    df_max['Last [tonn]'] = df_max['Last [N]'] / (g * 1000)
    df_max['Maks vertikal last [tonn]'] = df_data.apply(lambda row: row['Z_forces'][row['Z_forces_argmax']], axis=1)
    df_max['Min vertikal last [tonn]'] = df_data.apply(lambda row: row['Z_forces'][row['Z_forces_argmin']], axis=1)
    df_max['Spenningsvidde [MPa]'] = df_data.apply(lambda row: row['Right_web'][row['Right_web_argmax']], axis=1)
    df_max['Konvergens'] = df_data.apply(lambda row: row['Conv_norm'][row['Conv_norm_argmax']], axis=1)
    
    if 'Force_indices' in data_dicts.keys(): # Enough to only check for Force_indices
        df_max['LT last'] = df_data.apply(lambda row: row['Force_indices'][row['Forces_argmax']], axis=1).astype(np.int64)
        df_max['LT maks vertikal'] = df_data.apply(lambda row: row['Z_forces_indices'][row['Z_forces_argmax']], axis=1).astype(np.int64)
        df_max['LT min vertikal'] = df_data.apply(lambda row: row['Z_forces_indices'][row['Z_forces_argmin']], axis=1).astype(np.int64)
        df_max['LT spenningsvidde'] = df_data.apply(lambda row: row['Right_web_indices'][row['Right_web_argmax']], axis=1).astype(np.int64)
        df_max['LT konvergens'] = df_data.apply(lambda row: row['Conv_norm_indices'][row['Conv_norm_argmax']], axis=1).astype(np.int64)
    
    if return_df_data:
        return df_max, df_data
    else:
        return df_max

def avz_to_df(avz_path, is_accident, is_nice=False):
    '''Get a complete DataFrame from .avz-file.'''
    data_dicts = collect_avz_data(avz_path, blocks)
    df_model = model(avz_path, is_accident, is_nice)
    df_result = avz_result(data_dicts)
    df_result['Utnyttelse [%]'] = df_result['Last [tonn]'] * 100 / df_model['Lastgrense [tonn]']
    return pd.merge(df_model, df_result, left_index=True, right_index=True)

def avz_env_mapping(avz_path):
    '''Create mapping from environment index back to file names.'''
    with zipfile.ZipFile(avz_path) as zfile:
        with zfile.open('model.avs') as file:
            is_inside = False
            file_names = []
            for line in file:
                nice_line = line.decode('Latin-1').strip()
                if (nice_line == '}') & is_inside: # End of block
                    break
                if is_inside:
                    file_names.append(nice_line)
                if nice_line == 'FILEINDEX {': # Start collecting after this line
                    is_inside = True
            
    mapping = {i: file_name for i, file_name in enumerate(reversed(file_names), 1)}
    return mapping

def make_env_bins(series, num_env=None, make_plot=True, figsize=(10,6)):
    env_bins = np.unique(series, return_counts=True)
    
    if make_plot:
        fig = plt.figure(figsize=figsize)
        plt.bar(env_bins[0],env_bins[1])
        if num_env == None:
            plt.xticks(np.arange(1, max(env_bins[0])+1))
        else:
            plt.xticks(np.arange(1, num_env+1))
        plt.xlabel('Lasttilfelle', size=15)
        plt.ylabel('Antall komponenter', size=15)
        plt.title('Dimensjonerende lasttilfeller', size=15)
        plt.show()
    
    return fig

def direction(degrees, numeric=True):
    intervals = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
    if numeric:
        interval_name = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        interval_name = ['N', 'NØ', 'Ø', 'SØ', 'S', 'SV', 'V', 'NV']
    for i in range(0,len(intervals)-1):
        if intervals[i] <= degrees < intervals[i+1]:
            return interval_name[i+1]
    return interval_name[0]


def read_key(key_path):
	'''Reads relevant data from key.txt-file.
	mass_w   ==> effective mass in water [kg]
	mass     ==> mass [kg]
	bouyancy ==> bouyancy [kg]
	length   ==> length [m]'''
	with open(key_path, "r") as file:
	    header = ['ID', 'mass_w', 'mass', 'boyancy', 'L']
	    lines = {name: [] for name in header}
	    for line in file:
	        if 'Component' in line:
	            data = line.split()
	            if len(data) == 6:
	                lines[header[0]].append(int(data[1]))
	                lines[header[-1]].append(float(data[-1]))
	                for i in range(1, len(header)-1):
	                    lines[header[i]].append(float(data[i+1])/g)
	    
	    df_key = pd.DataFrame(lines)
	    df_key.set_index('ID', inplace=True)
	    return df_key
