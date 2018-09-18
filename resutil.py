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

xml_header = [
    'load_limit', 'edit_id', 'materialcoeff',
    'mbl', 'name', 'component', 'material', 'id'
]
result_header = [
    'force', 'load', 'max_zload',
    'min_zload', 'right_web', 'conv_norm',
    'force_index', 'max_zload_index', 'min_zload_index',
    'right_web_index', 'conv_norm_index'
]
final_header = [
    'Lastgrensen [tonn]', 'Edit ID', 'Materialkoeffisient',
    'MBL [tonn]', 'Navn', 'Komponent', 'Materiale', 'Gruppe',
    'Kraft [N]', 'Last [tonn]', 'Maks vertikal last [tonn]',
    'Min vertikal last [tonn]', 'Spenningsvidde [MPa]',
    'Konvergens', 'LT Last', 'LT maks vertikal last',
    'LT min vertikal last', 'LT spenningsvidde',
    'LT konvergens', 'Utnyttelse [%]', 'MBL krav [tonn]'
]

def model(path, is_accident, is_nice=False, to_clipboard=False):
    '''is_nice is true when names are clever (or nice!). 
    When is_nice=True components are categorized.
    When to_clipboard is true MBL, Materialcoeff and Materials 
    are written to clipboard.'''
    
    if path[-4:] == '.avz':        
        with zipfile.ZipFile(file=path) as zfile:
            with zfile.open('model.xml') as file:
                root = et.XML(file.read().decode('Latin-1'))
    elif path[-4:] == '.xml':
        tree = et.parse('Testmiljø/Horsvaagen_90m_.xml')
        root = tree.getroot()
    else:
        print('Input file must be either .avz or .xml.')
        return None
    
    model = {header: [] for header in xml_header}
    for comp in root.iter('component'):
        mcoeff      = float(comp.attrib['materialcoeff'])
        mbl         = float(comp.attrib['breakingload'])/(g*1000)
        name        = comp.attrib['name']
        name_list   = name.split(':')
        model[xml_header[5]].append(name_list[0].strip())   # component
        model[xml_header[6]].append(name_list[-1].strip())  # material
        model[xml_header[4]].append(name)                   # name
        model[xml_header[2]].append(float(comp.attrib['materialcoeff'])) # materialcoeff
        model[xml_header[3]].append(mbl)                    # mbl
        model[xml_header[7]].append(int(comp.attrib['id'])) # id
        model[xml_header[1]].append(int(comp.attrib['number'])) # edit_id
        if is_accident:
            model[xml_header[0]].append(mbl/(mcoeff/1.5))   # load_limit
        else:
            model[xml_header[0]].append(mbl/(mcoeff*1.15))  # load_limit
    
    df_model = pd.DataFrame(data = model)
    df_model.set_index(xml_header[7], inplace=True)
    
    if is_nice:
        df_model[[xml_header[5], 'group']] = df_model[xml_header[5]].str.split('_', expand=True)
        df_model.loc[:, 'group'] = df_model['group'].astype('category')
    
    return df_model

def collect_avz_data(avz_path, blocks):
    '''Parse data from file to a dictionary.'''
    with zipfile.ZipFile(avz_path) as zfile:
        with zfile.open('model.avs') as file:
            is_inside = False
            content = []
            data_dicts = {}
            data_key = ''
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
    df_max.index.name = 'id'
    df_max[result_header[0]] = df_data.apply(lambda row: row['Forces'][row['Forces_argmax']], axis=1) # force
    df_max[result_header[1]] = df_max[result_header[0]] / (g * 1000) # load
    df_max[result_header[2]] = df_data.apply(lambda row: row['Z_forces'][row['Z_forces_argmax']], axis=1) / (g * 1000) # max_zload
    df_max[result_header[3]] = df_data.apply(lambda row: row['Z_forces'][row['Z_forces_argmin']], axis=1) / (g * 1000) # min_zload
    df_max[result_header[4]] = df_data.apply(lambda row: row['Right_web'][row['Right_web_argmax']], axis=1) # right_web
    df_max[result_header[5]] = df_data.apply(lambda row: row['Conv_norm'][row['Conv_norm_argmax']], axis=1) # conv_norm
    
    if 'Force_indices' in data_dicts.keys(): # Enough to only check for Force_indices
        df_max[result_header[6]] = df_data.apply(lambda row: row['Force_indices'][row['Forces_argmax']], axis=1).astype(np.int64) # force_index
        df_max[result_header[7]] = df_data.apply(lambda row: row['Z_forces_indices'][row['Z_forces_argmax']], axis=1).astype(np.int64) # max_zload_index
        df_max[result_header[8]] = df_data.apply(lambda row: row['Z_forces_indices'][row['Z_forces_argmin']], axis=1).astype(np.int64) # min_zload_index
        df_max[result_header[9]] = df_data.apply(lambda row: row['Right_web_indices'][row['Right_web_argmax']], axis=1).astype(np.int64) # right_web_index
        df_max[result_header[10]] = df_data.apply(lambda row: row['Conv_norm_indices'][row['Conv_norm_argmax']], axis=1).astype(np.int64) # conv_norm_index
    
    if return_df_data:
        return df_max, df_data
    else:
        return df_max

def avz_to_df(avz_path, is_accident, is_nice=False):
    '''Get a complete DataFrame from .avz-file.'''
    data_dicts = collect_avz_data(avz_path, blocks)
    df_model = model(avz_path, is_accident, is_nice)
    df_result = avz_result(data_dicts)
    df_result['utilization'] = df_result[result_header[1]] * 100 / df_model[xml_header[0]]
    if is_accident:
        df_result['mbl_bound'] = df_result[result_header[0]] * df_model[xml_header[2]] / (1.5 * g * 1000)
    else:
        df_result['mbl_bound'] = df_result[result_header[0]] * df_model[xml_header[2]] * (1.15 / (g * 1000))
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

def summarize(df_list, ref_list):
    '''Compares results pairwise.
    Returns indexed summary from both.'''
    assert len(df_list) == len(ref_list), 'Input lists must have same length.'

    df1 = df_list.pop(0)
    ref1 = ref_list.pop(0)
    df_final = df1.copy(deep=True)
    force_columns = (result_header[:2]
                     + [result_header[5], 'utilization', 'mbl_bound'])

    for df, ref in zip(df_list, ref_list):
        # Filters
        is_more_utilized = df['utilization'] > df_final['utilization']
        is_bigger_vertical_max = df[result_header[2]] > df_final[result_header[2]] # max_zload
        is_bigger_vertical_min = df[result_header[3]] > df_final[result_header[3]] # min_zload
        #is_less_conv = df[result_header[5]] > df_final[result_header[5]] # conv_norm
        # Update values
        df_final.loc[is_more_utilized, force_columns] = df.loc[is_more_utilized, force_columns] # force dependent columns
        df_final.loc[is_bigger_vertical_max, result_header[2]] = df.loc[is_bigger_vertical_max, result_header[2]] # max_zload
        df_final.loc[is_bigger_vertical_min, result_header[3]] = df.loc[is_bigger_vertical_min, result_header[3]] # min_zload
        # Update indices
        df_final.loc[is_more_utilized, result_header[6]] = ref # force_index and conv_norm_index
        df_final.loc[is_more_utilized, result_header[10]] = ref
        df_final.loc[is_bigger_vertical_max, result_header[7]] = ref # max_zload_index
        df_final.loc[is_bigger_vertical_min, result_header[8]] = ref # min_zload_index

    index_columns = [result_header[6],
                    result_header[7],
                    result_header[8],
                    result_header[10]]
    df_final.loc[:, index_columns] = df_final.loc[:, index_columns].fillna(ref1) # Only necessary if first df is not intact state
    return df_final

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
    with open(key_path, 'r') as file:
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

def collect_env(avz_path):
    '''Read environment data from .avz-file and return
    it in a dictionary.'''
    with zipfile.ZipFile(avz_path) as zfile:
        with zfile.open('model.xml') as file:
            root = et.XML(file.read().decode("Latin-1"))

    current_keys = ['velocity_5', 'direction_5', 'velocity_15', 'direction_15']
    load_keys = root[0][0].keys() + current_keys
    keys_to_numeric = ['waveamplitude', 'waveperiod', 'waveangle',
                       'wavetype', 'currentx', 'currenty', 'windx', 'windy']
    env_data = {key: [] for key in load_keys}

    for load in root[0]:
        current1 = load[0][0]
        current2 = load[0][1]
        for key in keys_to_numeric:
            env_data[key].append(float(load.attrib[key]))
        env_data['group'].append(int(load.attrib['group']))
        env_data['type'].append(load.attrib['type'])
        env_data[current_keys[0]].append(float(current1.attrib["velocity"]))
        env_data[current_keys[1]].append(float(current1.attrib["direction"]))
        env_data[current_keys[2]].append(float(current2.attrib["velocity"]))
        env_data[current_keys[3]].append(float(current2.attrib["direction"]))
    return env_data

def read_env_data(env_data):
    '''Treat environment data from dictionary
    and return it in a DataFrame.'''
    df_env = pd.DataFrame(env_data)
    df_env.type = pd.Categorical(df_env.type)
    df_env.index += 1
    df_env.waveamplitude = df_env.waveamplitude * 1.05 # For some reason
    df_env['wind'] = np.sqrt(df_env.windx**2 + df_env.windy**2)
    df_env['wind_direction'] = (np.arctan2(df_env.windx, df_env.windy)
                              * 180 / np.pi + 180)
    df_env['current_5_direction'] = (np.arctan2(df_env.windx, df_env.windy)
                              * 180 / np.pi + 180)
    df_env['sector'] = df_env.wind_direction.apply(lambda r: direction(r, numeric=False))
    df_env['num_sector'] = df_env.wind_direction.apply(lambda r: direction(r, numeric=True))
    df_env['type'] = pd.Categorical(df_env['type'])
    return df_env[["sector", "waveamplitude", "waveperiod", "wind", "wind_direction",
                   "velocity_5", "direction_5", "velocity_15", "direction_15",
                  'type', 'group', 'num_sector', 'waveangle']]

def avz_to_env(avz_path):
    return read_env_data(collect_env(avz_path))