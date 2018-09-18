import numpy as np
import pandas as pd
from os import mkdir, listdir
import xml.etree.ElementTree as et
from scipy.constants import pi, g
import matplotlib.pyplot as plt
plt.style.use('bmh')

def test_col_names(material_df):
    '''Checks that the header is as expected.'''
    header = ['Diameter [mm]',
              'Massetetthet [kg/m]',
              'MBL [tonn]',
              'MBL [kN]',
              'E-modul [Pa]',
              'Materialkoeffisient'] # Expected header
    for col_name in header:
        assert col_name in material_df.columns, \
        '{} not in dataset.'.format(col_name)

def test_col_types(material_df):
    '''Checks that all columns are float64.'''
    col_types = material_df.dtypes
    for i, col_type in enumerate(col_types):
        assert col_type == np.float64, \
        '{} is not float64.'.format(material_df.columns[i])

def test_increasing_observations(material_df):
    '''Checks that all, except last two columns, are 
    increasing in value with each observation.'''
    rows, cols = material_df.shape
    for j in range(cols-2):
        for i in range(rows-1):
            assert material_df.iloc[i, j] < material_df.iloc[i+1, j], \
            '{} is not increasing with each observation.'.format(material_df.columns[j])

def test_interdepence(material_df):
    '''Checks that MBL [tonn] and MBL [kN]
    corresponds within tolerance of 1000 N.'''
    tolerance = 1.0 # kN
    low_bound = (material_df['MBL [kN]'] - tolerance
                 <= material_df['MBL [tonn]'] * g)
    up_bound = (material_df['MBL [kN]'] + tolerance\
                >= material_df['MBL [tonn]'] * g)
    bounds = ~(low_bound & up_bound)
    assert low_bound.all() and up_bound.all(), \
    'Discrepancy between MBL [tonn] and MBL [kN] in \n{}'.format(material_df[bounds])

def test_material(material_df):
    '''Runs all tests.'''
    test_col_names(material_df)
    test_col_types(material_df)
    test_increasing_observations(material_df)
    test_interdepence(material_df)

def make_control_figures(material_df, fig_name, folder=None):
    '''Makes figures that are important for inspection.'''
    material_df_mod = material_df.set_index("Diameter [mm]")
    min_D = np.floor(material_df["Diameter [mm]"].min() / 10) * 10
    max_D = np.ceil(material_df["Diameter [mm]"].max() / 10 + 1) * 10
    fig1 = material_df_mod.plot(subplots=True,
                                figsize=(10,20),
                                xticks=np.arange(min_D, max_D, 10),
                                marker="o",
                                title=fig_name,
                               fontsize=20)
    if folder:
        plt.savefig(folder + "/" + fig_name + ".svg", format="svg")
        plt.savefig(folder + "/" + fig_name + ".png", format="png")
    else:
        plt.savefig(fig_name + ".svg", format="svg")
        plt.savefig(fig_name + ".png", format="png")
    plt.clf()
    fig2 = material_df_mod.diff().plot(subplots=True,
                                       figsize=(10,20),
                                       xticks=np.arange(min_D, max_D, 10),
                                       marker="o",
                                       title=fig_name,
                                      fontsize=20)
    if folder:
        plt.savefig(folder + "/" + fig_name + "_diff.svg", format="svg")
        plt.savefig(folder + "/" + fig_name + "_diff.png", format="png")
    else:
        plt.savefig(fig_name + "_diff.svg", format="svg")
        plt.savefig(fig_name + "_diff.png", format="png")
    plt.clf()

def make_xml_library(material_df, material_name, material_suffix, is_chain, folder=None):
    """Reads from material DataFrame and
    writes readable library of steel chain for AquaEdit."""
    sea_density = 1025.0
    steel_density = 7850.0 # value used by AS
    mass_correction = ((steel_density - sea_density)
                       / steel_density) # correction for buoyancy of steel in sea water
    
    for index, row in material_df.iterrows():
        full_name = str(int(row["Diameter [mm]"])) + " " + material_suffix
        description = str(int(row["Diameter [mm]"])) + " mm " + material_name
        radius = row["Diameter [mm]"]  / 2e3 # meters
        mass = row["Massetetthet [kg/m]"] # per meter
        if is_chain:
            addedmassz = 1.0
            addedmassy = 1.0
            area = 2 * pi * radius ** 2
            rho = mass / area # mass density (kg/m^3)
            mass_water = mass * mass_correction
            volume = mass / steel_density
        else:
            addedmassz = 0.0
            addedmassy = 0.0
            area = pi * radius ** 2
            rho = mass / area # mass density (kg/m^3)
            mass_water = 0.001
            volume = area
        
        library = et.Element("Library")
        truss = et.SubElement(library, "truss", 
                              id = str(index + 1), 
                              name = full_name)
        et.SubElement(truss, "description", des=description)
        et.SubElement(truss, "mooring",
                      pretension = "0.0",
                      addedmasscoefflocalz = str(addedmassz),
                      addedmasscoefflocaly = str(addedmassy),
                      massdensity = str(rho),
                      young = str(row["E-modul [Pa]"]),
                      noCompressionForces = "false",
                      volumeoverwritten = "true",
                      volume = str(volume),
                      areal = str(area),
                      weightinwateroverwritten = "true",
                      weightInWater = str(mass_water),
                      weightInAir = str(mass))
        et.SubElement(truss, "extra",
                      trusstype = "3", # custom type
                      materialcoefficient = str(row["Materialkoeffisient"]),
                      breakingload = str(row["MBL [kN]"] * 1000))
        et.SubElement(truss, "loadmodel",
                      dragCoeffy = "1.2",
                      dragCoeffz = "1.2",
                      dragArealy = str(row["Diameter [mm]"] / 1000),
                      dragArealyz = str(row["Diameter [mm]"] / 1000),
                      constructionDamping = "0.0",
                      rayleighStiffness = "0.0",
                      tangentialDragCoeff = "0.0",
                      numvelocities = "0",
                      hullnumPoints = "0",
                      closeSurfaceNumPoints = "0",
                      numWaveHeading = "0",
                      viscousRollDamping = "0.0",
                      massRadius = "0.0",
                      LoadType = "MORRISON")
        
        tree = et.ElementTree(library)
        if folder:
            if folder in listdir():
                tree.write(folder + "/" + full_name + ".xml")
            else:
                mkdir(folder)
                tree.write(folder + "/" + full_name + ".xml")
        else:
            tree.write(full_name + ".xml")

def make_library(lib_path, material_name, material_suffix, sheet_name, is_chain, folder=None):
    '''Runs test battery, creates figures for inspection and AE friendly library.'''
    material_df = pd.read_excel(lib_path, sheet_name=sheet_name, dtype=float)
    make_control_figures(material_df, fig_name=sheet_name, folder="Figurer")
    test_material(material_df)
    make_xml_library(material_df, material_name, material_suffix, is_chain, folder)

def main():
  import sys
  filepath = sys.argv[1]
  mat_dict = {
#   Suffix: [material_name, sheet_name, is_chain]
    "STec-3": ["SuperTec 3-slått", "3-SuperTec", False],
    "STec-8": ["SuperTec 8-slått", "8-SuperTec", False],
    "Sdan-3": ["Superdan 3-slått", "3-Superdan", False],
    "Sdan-8": ["Superdan 8-slått", "8-Superdan", False],
    "Nyl-3": ["Nylon 3-slått", "3-Nylon", False],
    "Nyl-8": ["Nylon 8-slått", "8-Nylon", False],
    "GS-3": ["Gold Safety 3-slått", "3-GoldSafety", False],
    "AlKj": ["Alloykjetting", "Alloykjetting", True],
    "AnKj": ["Ankerkjetting", "Ankerkjetting", True]
    }

  for suffix, mat_list in mat_dict.items():
      print(mat_list[1]) # sheet_name
      make_library(lib_path=filepath,
                  material_name=mat_list[0],
                  material_suffix=suffix,
                  sheet_name=mat_list[1],
                  folder=mat_list[1],
                  is_chain=mat_list[2])

if __name__ == '__main__':
  main()