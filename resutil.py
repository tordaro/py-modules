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
    'STRESS_LINE_LIST:Local_section_forces.Max_axial_force_[N] {', 
    'STRESS_LINE_LIST:Local_section_forces.Max_axial_force_[N]_INDEX {',
    'STRESS_LINE_LIST:Global_section_forces.Max_force_Z_[N] {',
    'STRESS_LINE_LIST:Global_section_forces.Max_force_Z_[N]_INDEX {',
    'STRESS_LINE_LIST:Nominal_stress_range.Right_web_[MPa] {',
    'STRESS_LINE_LIST:Nominal_stress_range.Right_web_[MPa]_INDEX {',
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

def map_header(df_result):
    '''Maps columns names from internal ones
    to pretty ones.'''
    internal_header = [
        'load_limit', 'edit_id', 'materialcoeff',
        'mbl', 'name', 'component', 'material', 'segment',
        'force', 'load', 'max_zload',
        'min_zload', 'right_web', 'conv_norm',
        'force_index', 'max_zload_index', 'min_zload_index',
        'right_web_index', 'conv_norm_index', 'utilization',
        'mbl_bound', 'mass_w', 'mass', 'boyancy', 'length',
        'force_source', 'min_zload_source','max_zload_source',
        'conv_norm_source', 'right_web_source'
    ]
    pretty_header = [
        'Lastgrense [tonn]', 'Edit ID', 'Materialkoeffisient',
        'MBL [tonn]', 'Navn', 'Komponent', 'Materiale', 'Segment',
        'Kraft [N]', 'Last [tonn]', 'Maks vertikal last [tonn]',
        'Min vertikal last [tonn]', 'Spenningsvidde [MPa]',
        'Konvergens', 'LT Last', 'LT maks vertikal last',
        'LT min vertikal last', 'LT spenningsvidde',
        'LT konvergens', 'Utnyttelse [%]', 'MBL krav [tonn]',
        'MBL-krav [tonn]', 'Effektiv masse [kg]',
        'Oppdrift [kg]', 'Lengde [m]', 'Makslast kilde',
        'Min vertikal last kilde', 'Maks vertikal last kilde',
        'Konvergens kilde', 'Spenningsvidde kilde'
    ]
    header_mapping = dict(zip(internal_header, pretty_header))
    df_pretty = df_result.copy()
    df_pretty.columns = df_result.columns.map(header_mapping)
    return df_pretty


def reorder_and_filter(df_result):
    '''Reorder the columns and exclude
    uninteresting ones.'''
    desired_order = [
        'component', 'segment', 'material',
        'materialcoeff', 'length', 'mass',
        'mbl', 'mbl_bound', 'load', 'load_limit',
        'utilization', 'min_zload', 'max_zload',
        'right_web', 'conv_norm', 'force_index', 
        'max_zload_index', 'min_zload_index',
        'right_web_index', 'conv_norm_index',
        'force_source', 'min_zload_source','max_zload_source',
        'conv_norm_source', 'right_web_source'
    ]
    allowed_headers = [name for name in desired_order if name in df_result.columns]
    return df_result[allowed_headers]


def _model(path, is_accident, is_nice=False, to_clipboard=False):
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
    
    xml_header = [
    'load_limit', 'edit_id', 'materialcoeff',
    'mbl', 'name', 'component', 'material', 'id'
    ]
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
        df_model[[xml_header[5], 'segment']] = df_model[xml_header[5]].str.split('_', expand=True)
        df_model.loc[:, 'segment'] = df_model['segment'].astype('category')
    
    return df_model


def _collect_avz_vertices(avz_path):
    '''Parse data from file to a dictionary.'''
    with zipfile.ZipFile(avz_path) as zfile:
        with zfile.open('model.avs') as file:
            is_inside = False
            positions = {"x": [],
                         "y": [],
                         "z": []}
            for line in file:
                nice_line = line.decode('Latin-1').strip()
                if '}' in nice_line and is_inside:
                    return positions

                if is_inside:
                    data_list = nice_line.split()
                    positions["x"].append(np.float64(data_list[2]))
                    positions["y"].append(np.float64(data_list[3]))
                    positions["z"].append(np.float64(data_list[4]))
                
                if "VERTEX_LIST {" == nice_line:
                    is_inside = True
    return positions


def _collect_avz_edges(avz_path):
    with zipfile.ZipFile(avz_path) as zfile:
            with zfile.open('model.avs') as file:
                is_inside = False
                edges = {}
                for line in file:
                    nice_line = line.decode('Latin-1').strip()
                    if "TIMESTEP {" == nice_line:
                        return edges

                    if '}' in nice_line and is_inside:
                        is_inside = False

                    if is_inside:
                        edge_list = nice_line.split()
                        edges[ID].append((np.int32(edge_list[-3]),
                                              np.int32(edge_list[-1])))

                    if 'LINE_LIST {' == nice_line:
                        is_inside = True
                        ID = str(next(file).decode('Latin-1').strip().split()[-1])
                        next(file)  # To skip LINE_THICKNESS
                        edges[ID] = []
    return edges


def _collect_avz_data(avz_path, blocks):
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


def _avz_result(data_dicts, return_df_data=False):
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
    df_max['force'] = df_data.apply(lambda row: row['Forces'][row['Forces_argmax']], axis=1)
    df_max['load'] = df_max['force'] / (g * 1000)
    df_max['max_zload'] = df_data.apply(lambda row: row['Z_forces'][row['Z_forces_argmax']], axis=1) / (g * 1000)
    df_max['min_zload'] = df_data.apply(lambda row: row['Z_forces'][row['Z_forces_argmin']], axis=1) / (g * 1000)
    df_max['right_web'] = df_data.apply(lambda row: row['Right_web'][row['Right_web_argmax']], axis=1)
    df_max['conv_norm'] = df_data.apply(lambda row: row['Conv_norm'][row['Conv_norm_argmax']], axis=1)
    
    if 'Force_indices' in data_dicts.keys(): # Enough to only check for Force_indices
        df_max['force_index'] = df_data.apply(lambda row: row['Force_indices'][row['Forces_argmax']], axis=1).astype(np.int64)
        df_max['max_zload_index'] = df_data.apply(lambda row: row['Z_forces_indices'][row['Z_forces_argmax']], axis=1).astype(np.int64)
        df_max['min_zload_index'] = df_data.apply(lambda row: row['Z_forces_indices'][row['Z_forces_argmin']], axis=1).astype(np.int64)
        df_max['right_web_index'] = df_data.apply(lambda row: row['Right_web_indices'][row['Right_web_argmax']], axis=1).astype(np.int64)
        df_max['conv_norm_index'] = df_data.apply(lambda row: row['Conv_norm_indices'][row['Conv_norm_argmax']], axis=1).astype(np.int64)
    
    if return_df_data:
        return df_max, df_data
    else:
        return df_max


def avz_to_df(avz_path, is_accident, is_nice=False):
    '''Get a complete DataFrame from .avz-file.'''
    data_dicts = _collect_avz_data(avz_path, blocks)
    df_model = _model(avz_path, is_accident, is_nice)
    df_result = _avz_result(data_dicts)
    df_result['utilization'] = df_result['load'] * 100 / df_model['load_limit']
    if is_accident:
        df_result['mbl_bound'] = df_result['force'] * df_model['materialcoeff'] / (1.5 * g * 1000)
    else:
        df_result['mbl_bound'] = df_result['force'] * df_model['materialcoeff'] * (1.15 / (g * 1000))
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
    '''Summarizes all results in df_list
    with correct indices, and sources from ref_list.'''
    assert len(df_list) == len(ref_list), 'Input lists must have same length.'

    df1 = df_list[0]
    ref1 = ref_list[0]
    df_final = df1.copy(deep=True)
    
    force_columns = ['force', 'load', 'load_limit',
                    'conv_norm', 'utilization', 'mbl_bound'] # Interdependent columns
    
    source_columns = ['force_source', 'min_zload_source',
                     'max_zload_source', 'conv_norm_source',
                     'right_web_source']
    # Add source columns
    for source in source_columns:
        df_final[source] = ref1
    
    is_max_out_final = 'force_index' in df_final.columns
    control_value = 1
    for df, ref in zip(df_list[1:], ref_list[1:]):
        is_max_out = 'force_index' in df.columns # Either all or none index columns are present
        # Filters
        is_more_utilized = df['utilization'] > df_final['utilization']
        is_bigger_zmin = df['min_zload'] > df_final['min_zload']
        is_bigger_zmax = df['max_zload'] > df_final['max_zload']
        is_bigger_rweb = df['right_web'] > df_final['right_web']
        # Update values
        df_final.loc[is_more_utilized, force_columns] = df.loc[is_more_utilized, force_columns]
        df_final.loc[is_bigger_zmax, 'max_zload'] = df.loc[is_bigger_zmax, 'max_zload']
        df_final.loc[is_bigger_zmin, 'min_zload'] = df.loc[is_bigger_zmin, 'min_zload']
        # Update sources
        df_final.loc[is_more_utilized, 'force_source'] = ref
        df_final.loc[is_more_utilized, 'conv_norm_source'] = ref
        df_final.loc[is_bigger_zmin, 'min_zload_source'] = ref
        df_final.loc[is_bigger_zmax, 'max_zload_source'] = ref
        # Update indices and rweb only if is_max_out
        if is_max_out and is_max_out_final:
            # Both df and df_final has force_indices
            df_final.loc[is_more_utilized, 'force_index'] = df.loc[is_more_utilized, 'force_index']
            df_final.loc[is_more_utilized, 'conv_norm_index'] = df.loc[is_more_utilized, 'conv_norm_index']
            df_final.loc[is_bigger_zmin, 'min_zload_index'] = df.loc[is_bigger_zmin, 'min_zload_index']
            df_final.loc[is_bigger_zmax, 'max_zload_index'] = df.loc[is_bigger_zmax, 'max_zload_index']
            
            df_final.loc[is_bigger_rweb, 'right_web_index'] = df.loc[is_bigger_rweb, 'right_web_index']
            df_final.loc[is_bigger_rweb, 'right_web'] = df.loc[is_bigger_rweb, 'right_web']
            df_final.loc[is_bigger_rweb, 'right_web_source'] = ref

        elif is_max_out:
            # Only df has force indices. Necessary if df_final is not a max_out result.
            # This block will only run once.
            assert control_value == 1,\
            'Naughty program trying to do an illegal operation. Not today!'
            df_final.loc[:, 'force_index'] = df.loc[:, 'force_index']
            df_final.loc[:, 'conv_norm_index'] = df.loc[:, 'conv_norm_index']
            df_final.loc[:, 'min_zload_index'] = df.loc[:, 'min_zload_index']
            df_final.loc[:, 'max_zload_index'] = df.loc[:, 'max_zload_index']
            
            df_final.loc[:, 'right_web_index'] = df.loc[:, 'right_web_index']
            df_final.loc[:, 'right_web'] = df.loc[:, 'right_web']
            df_final.loc[:, 'right_web_source'] = ref

            is_max_out_final = True
            control_value += 1

    return df_final

def add_indices(df_result, index, inplace=True):
    'Add or change index columns. Inplace as default.'
    index_columns = ['force_index',
                    'max_zload_index',
                    'min_zload_index',
                    'right_web_index',
                    'conv_norm_index']
    if inplace:    
        for index_col in index_columns:
            df_result.loc[:, index_col] = index
    else:
        df_result_copy = df_result.copy(deep=True)
        for index_col in index_columns:
            df_result_copy.loc[:, index_col] = index
        return df_result_copy


def make_env_bins(series, num_env=None, make_plot=True, figsize=(10,6)):
    print('WARNING: Deprecated function. Replaced by lt_summary().')
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


def categorize_by_id(id_list, segment_names):
    '''Function that categorizes all components
    by given ID interval.'''
    assert len(segment_names) == len(id_list) - 1,\
    'id_list has {} elements, should have {} elements.'.format(len(id_list), len(segment_names)+1)
    
    segments = []
    for i in range(len(segment_names)):
        segments.extend([segment_names[i]] * (id_list[i+1] - id_list[i]))
    
    return segments


def read_key(key_path):
    '''Reads relevant data from key.txt-file.
    mass_w   ==> effective mass in water [kg]
    mass     ==> mass [kg]
    bouyancy ==> bouyancy [kg]
    length   ==> length [m]'''
    with open(key_path, 'r') as file:
        header = ['ID', 'mass_w', 'mass', 'boyancy', 'length']
        lines = {name: [] for name in header}
        for line in file:
            if ' Mass centre beams and trusses ' in line:
                # To stop after first block is read
                break
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


def _collect_env(avz_path):
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
    return read_env_data(_collect_env(avz_path))


def lt_summary(df_list, ref_list, num_lt=16, plot=True, figsize=(16,10)):
    '''Make barplot of force index from each df.
    If plot is False, a DataFrame with the summary
    is returned.'''
    indices = pd.DataFrame(np.zeros((num_lt, len(df_list))))
    indices.index += 1
    indices.columns = ref_list
    for df, ref in zip(df_list, ref_list):
        bins = np.unique(df.force_index, return_counts=True)
        indices.loc[bins[0] ,ref] = bins[1]
    
    if plot:
        indices.plot.bar(figsize=figsize, rot=0)
        plt.xlabel("Lasttilfelle", size=15)
        plt.ylabel("Komponenter", size=15)
        plt.savefig("../Figurer/lt.png", format="png")
        plt.show()
    else:
        return indices


def prioritize_components(result_df, by, n_components, segments=slice(None)):
    '''Prioritize n_components components in results_df in segment
    groups by by-parameter. Segments of interest are given by segments.
    Convenient for making material configuration table.'''
    ids = result_df.groupby('segment')[by].nlargest(n_components).reset_index().iloc[:,1]
    prioritized = result_df.loc[ids].round(1).set_index(['segment', by])
    return reorder_and_filter(prioritized.loc[segments])


def pivot_config(result_df, segments, columns=['material', 'length'], key_path=None):
    '''Merge result_df and data from key_path, filter results by comp_filter,
    reorder df by components and segments and choose data blocks given by
    column blocks.'''
    segment_filter = (result_df.segment == segments[0])
    for i in range(1, len(segments)):
        segment_filter = segment_filter | (result_df.segment == segments[i])
    
    if key_path:
        key_df = read_key(key_path)
        result_key = pd.merge(result_df, key_df, left_index=True, right_index=True)
        result_slice = result_key.loc[segment_filter]
        return result_slice.pivot(index='component', columns='segment', values=columns)
    else:
        result_slice = result_df.loc[segment_filter]
        return result_slice.pivot(index='component', columns='segment', values=columns)


def make_conv_df(conv_paths):
    '''Collects iteration steps for all load steps
    in files given by the file list conv_paths.
    Returns DataFrame with each loadstep as an
    observation, and files as variables.
    Recommended combine this funciton with
    glob("../**/*conv.txt", recursive=True).'''
    conv_data = {}
    for conv_path in conv_paths:
        conv_data[conv_path] = []
        with open(conv_path) as conv_file:
            for line in conv_file:
                if 'CONVERGENCE' in line:
                    nice_list = line.strip().split()
                    # List index 4 gives iteration step also when
                    # convergence is not achieved
                    conv_data[conv_path].append(float(nice_list[4]))
    
    # Transpose trick to fill shorter lists with NaNs
    conv_df = pd.DataFrame.from_dict(conv_data, orient='index').T
    return conv_df


def components_by_material(result):
    '''
    Lists components by materials, given materials in result.
    '''
    header = ["Materiale", "Komponent", "Lengder [m]", "Total lengde [m]"]
    mat_comp_list = []
    for material in result.material.unique():
        mat_filter = result.material == material
        comp_list = result.loc[mat_filter, "component"].astype(str).tolist()
        len_list = result.loc[mat_filter, "length"].round(1).astype(str).tolist()
        tot_len = result.loc[mat_filter, "length"].sum()
        mat_comp_list.append(
        	(material,
        	'; '.join(comp_list),
        	'; '.join(len_list),
        	round(tot_len,1))
        )

    mat_df = pd.DataFrame(mat_comp_list, columns=header)
    mat_df.set_index("Materiale", inplace=True)
    return mat_df
