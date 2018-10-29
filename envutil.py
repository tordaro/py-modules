# -*- coding: utf-8 -*-
"""
@author: tordaronsen
"""
import resutil
import pandas as pd
import numpy as np
import tabula as tb
from time import time
from scipy.constants import pi, g

def get_header(flavor):
    '''Get new header with a 
    different flavor.'''
    if flavor == "practical":
        practical_header = [
            "sektor",
            "hs",
            "tp",
            "vind",
            "vindretn",
            "strom5",
            "strom5retn",
            "strom15",
            "strom15retn",
            "steilhet"
        ]
        return practical_header
    elif flavor == "nice":
        nice_header = [
            'Sektor',
            'Hs [m]',
            'Tp [s]',
            'Vind [m/s]',
            'Vind [\N{DEGREE SIGN}]',
            'Strøm 5 m [m/s]',
            'Strøm 5 m [\N{DEGREE SIGN}]',
            'Strøm 15 m [m/s]',
            'Strøm 15 m [\N{DEGREE SIGN}]',
            'Steilhet'
        ]
        return nice_header
    elif flavor == "latex":
        latex_header = [
            'Sektor',
            '$H_s$ [m]',
            '$T_p$ [s]',
            '$U_r$ [m/s]',
            '$U_{\\theta}$ [$^\\circ$]',
            '$V_{5,r}$ [m/s]',
            '$V_{5,\\theta}$ [$^\\circ$]',
            '$V_{15,r}$ [m/s]',
            '$V_{15,\\theta}$ [$^\\circ$]',
            'Steilhet'
        ]
        return latex_header

def collect_APenv_data(path, num_cols, encoding="Latin-1"):
    '''Reads Akvaplan-Niva environmental data from .txt-file,
    and returns a dictionary with the data. The first num_cols
    should be the headers.'''
    data = {}
    header = []
    with open(path, "rb") as file:
        for i in range(num_cols):
            nice_line = file.readline().decode(encoding).strip()
            header.append(nice_line)
            data[nice_line] = []
        
        for i, line in enumerate(file):
            nice_line = line.decode(encoding).strip()
            data[header[i%num_cols]].append(float(nice_line))
    return data

def make_env_AP(path, decimal=b',', col_names=None):
    '''Reads Akvaplan-niva environmental data from 
    .csv-file, without headers, and returns it in 
    properly formatted DataFrame.'''
    AP_env = pd.read_csv(path, decimal=decimal, header=None)
    if not col_names:
        col_names = ['retning_strom',
                     '_målt_5', '_strom_5_10', '_strom_5_50',
                     'justert_5_10', 'justert_5_50',
                     '_målt_15', '_strom_15_10', '_strom_15_50',
                     'justert_15_10', 'justert_15_50',
                     'retning_vind',
                     'vind_10', 'vind_50',
                     'Hs_10', 'Tp_10',
                     'Hs_50', 'Tp_50']
    AP_env.columns = col_names

    AP_env["himmelretning"] = AP_env["retning_vind"].apply(resutil.direction)
    AP_env["sektor"] = AP_env["retning_vind"].apply(lambda r: resutil.direction(r, numeric=False))
    idx_df = AP_env.groupby("himmelretning").idxmax()

    env105050 = pd.DataFrame({
        "sektor": AP_env["sektor"][idx_df["Hs_10"]].values,
        "hs": AP_env["Hs_10"][idx_df["Hs_10"]].values,
        "tp": AP_env["Tp_10"][idx_df["Hs_10"]].values,
        "vind": AP_env["vind_10"][idx_df["Hs_10"]].values,
        "vindretn": AP_env["retning_vind"][idx_df["Hs_10"]].values,
        "strom5": AP_env["justert_5_50"][idx_df["justert_5_50"]].values / 100,
        "strom5retn": AP_env["retning_strom"][idx_df["justert_5_50"]].values,
        "strom15": AP_env["justert_15_50"][idx_df["justert_15_50"]].values / 100,
        "strom15retn": AP_env["retning_strom"][idx_df["justert_15_50"]].values
    })

    env501010 = pd.DataFrame({
        "sektor": AP_env["sektor"][idx_df["Hs_50"]].values,
        "hs": AP_env["Hs_50"][idx_df["Hs_50"]].values,
        "tp": AP_env["Tp_50"][idx_df["Hs_50"]].values,
        "vind": AP_env["vind_50"][idx_df["Hs_50"]].values,
        "vindretn": AP_env["retning_vind"][idx_df["Hs_50"]].values,
        "strom5": AP_env["justert_5_10"][idx_df["justert_5_10"]].values / 100,
        "strom5retn": AP_env["retning_strom"][idx_df["justert_5_10"]].values,
        "strom15": AP_env["justert_15_10"][idx_df["justert_15_10"]].values / 100,
        "strom15retn": AP_env["retning_strom"][idx_df["justert_15_10"]].values
    })

    if "Hs_10_hav" in AP_env:
        idx_df_hav = idx_df.dropna()

        env105050_hav = pd.DataFrame({
            "sektor": AP_env["sektor"][idx_df_hav["Hs_10_hav"]].values,
            "hs": AP_env["Hs_10_hav"][idx_df_hav["Hs_10_hav"]].values,
            "tp": AP_env["Tp_10_hav"][idx_df_hav["Hs_10_hav"]].values,
            "vind": AP_env["vind_10"][idx_df_hav["Hs_10_hav"]].values,
            "vindretn": AP_env["retning_vind"][idx_df_hav["Hs_50_hav"]].values,
            "strom5": AP_env["justert_5_50"][idx_df_hav["justert_5_50"]].values / 100,
            "strom5retn": AP_env["retning_strom"][idx_df_hav["justert_5_50"]].values,
            "strom15": AP_env["justert_15_50"][idx_df_hav["justert_15_50"]].values / 100,
            "strom15retn": AP_env["retning_strom"][idx_df_hav["justert_15_50"]].values
        })

        env501010_hav = pd.DataFrame({
            "sektor": AP_env["sektor"][idx_df_hav["Hs_50_hav"]].values,
            "hs": AP_env["Hs_50_hav"][idx_df_hav["Hs_50_hav"]].values,
            "tp": AP_env["Tp_50_hav"][idx_df_hav["Hs_50_hav"]].values,
            "vind": AP_env["vind_50"][idx_df_hav["Hs_50_hav"]].values,
            "vindretn": AP_env["retning_vind"][idx_df_hav["Hs_50_hav"]].values,
            "strom5": AP_env["justert_5_10"][idx_df_hav["justert_5_10"]].values / 100,
            "strom5retn": AP_env["retning_strom"][idx_df_hav["justert_5_10"]].values,
            "strom15": AP_env["justert_15_10"][idx_df_hav["justert_15_10"]].values / 100,
            "strom15retn": AP_env["retning_strom"][idx_df_hav["justert_15_10"]].values
        })

        env105050_hav.index += env501010.index[-1] + 1
        env501010_hav.index += env105050_hav.index[-1] + 1
        env501010 = pd.concat([env501010, env105050_hav, env501010_hav])

    env105050.index += 1
    env501010.index += 9
    env_final = pd.concat([env105050, env501010])
    env_final["steilhet"] = (env_final["tp"]**2 / env_final["hs"]) * (g / (1.9 * 2 * pi))
    return env_final

def _init_mc_current():
    '''Return initialized dictionary.
    Ready to be loaded with data.'''
    mc_current = {
        "strom5": np.zeros(16),
        "strom5retn": np.zeros(16),
        "strom15": np.zeros(16),
        "strom15retn": np.zeros(16)
    }
    return mc_current
    

def _load_mc_current(path, depth, data_dict):
    '''Load given initialized data_dict at 
    given depth.'''
    with open(path, "rb") as file:
        for i, line in enumerate(file):
            nice_list = line.decode("CP1252").split()

            data_dict["strom{}".format(depth)][(i+4) % 8] = float(nice_list[-1])
            data_dict["strom{}retn".format(depth)][(i+4) % 8] = float(nice_list[1])

            data_dict["strom{}".format(depth)][(i+4) % 8 + 8] = float(nice_list[-2])
            data_dict["strom{}retn".format(depth)][(i+4) % 8 + 8] = float(nice_list[1])


def _read_mc_waves(path):
    '''Read MultiConsult wave data and
    return it in properly formatted dictionary,
    ready to be DataFramed.'''
    mc_wave_data = {}
    with open(path, "rb") as file:
        for line in file:
            nice_list = line.decode("CP1252").split()
            mc_wave_data[nice_list[0]] = [float(val) for val in nice_list[1:]]

    mc_waves = {
        "sektor": np.array([resutil.direction(val, False) 
            for val in mc_wave_data["retning_vind_10"] + mc_wave_data["retning_vind_10"]]),
        "hs": np.array(mc_wave_data["Hs_10"] + mc_wave_data["Hs_50"]),
        "tp": np.array(mc_wave_data["Tp_10"] + mc_wave_data["Tp_50"]),
        "vind": np.array(mc_wave_data["vind_10"] + mc_wave_data["vind_50"]),
        "vindretn": np.array(mc_wave_data["retning_vind_10"] + mc_wave_data["retning_vind_50"])
    }
    return mc_waves


def _read_mc_ocean_waves(path, env_df):
    '''Read ocean MultiConsult ocean wave data
    and return DataFrame slice that can be merged
    with env_df.'''
    mc_ocean_data = {}
    with open(path, "rb") as file:
        for line in file:
            nice_list = line.decode("CP1252").split()
            mc_ocean_data[nice_list[0]] = [float(val) for val in nice_list[1:]]
    
    lt = ([int(val) for val in mc_ocean_data["lt_10"]]
          + [int(val) for val in mc_ocean_data["lt_50"]])
    mc_ocean_waves = {
        "hs": np.array(mc_ocean_data["Hs_10"] + mc_ocean_data["Hs_50"]),
        "tp": np.array(mc_ocean_data["Tp_10"] + mc_ocean_data["Tp_50"])
    }
    
    ocean_df = env_df.loc[lt]
    ocean_df.loc[lt, "hs"] = mc_ocean_waves["hs"]
    ocean_df.loc[lt, "tp"] = mc_ocean_waves["tp"]
    ocean_df.reset_index(inplace=True, drop=True)
    ocean_df.index += env_df.index[-1] + 1
    return ocean_df


def make_env_mc(waves_path, current_path_1, current_path_2, current_depths=[5,15], ocean_path=None):
    '''Make complete environmental DataFrame from MultiConsult
    data. Must have 3-4 input text files, in a standard format.'''
    mc_waves = _read_mc_waves(waves_path)
    
    mc_current = _init_mc_current()
    _load_mc_current(current_path_1, current_depths[0], mc_current)
    _load_mc_current(current_path_2, current_depths[1], mc_current)
    
    env_final = pd.DataFrame({**mc_waves, **mc_current})
    env_final.index += 1
    env_final["steilhet"] = (env_final["tp"]**2 / env_final["hs"]) * (g / (1.9 * 2 * pi))
    
    if ocean_path:
        ocean_env = _read_mc_ocean_waves(ocean_path, env_final)
        ocean_env["steilhet"] = (ocean_env["tp"]**2 / ocean_env["hs"]) * (g / (1.9 * 2 * pi))
        return pd.concat([env_final, ocean_env])
    else:
        return env_final

def ae_input(env_df):
    '''Take treated environment
    and return AE friendly environment.'''
    ae_env = env_df.copy(deep=True)
    ae_env["sys_heading"] = 90
    order = [
        "sys_heading",
        "hs",
        "tp",
        "vindretn",
        "strom5",
        "strom5retn",
        "vind",
        "strom15",
        "strom15retn"
    ]
    return ae_env[order]