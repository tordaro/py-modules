import gzip as gz
import pandas as pd
import folium as fl
from geographiclib.geodesic import Geodesic


def _format_degree_minutes(latitude, longitude):
    '''Format from decimal minutes to degrees decimal minutes.'''
    h_lat = int(latitude / 60)
    # The adding of 1e-4 and modulo 1e-3 business is to fix
    # the (really weird) olex round off. Olex rounds off at 9, not 5.
    min_lat = latitude - h_lat * 60 + 1e-4
    min_lat -= min_lat % 1e-3
    h_long = int(longitude / 60)
    min_long = longitude - h_long * 60 + 1e-4
    min_long -= min_long % 1e-3
    return '{:02d}\N{degree sign}{:06.3f} N - {:02d}\N{degree sign}{:06.3f} Ø'\
        .format(h_lat, min_lat, h_long, min_long)


def _read_olex_object_export(path):
    '''Reads olex file from given path
    and returns a dictionary with the data.'''
    header = ["given_name",
              "latitude",
              "longitude",
              "block",
              "timestamp",
              "formatted"]
    data = {name: [] for name in header}
    block_id = 1

    with gz.open(path) as file:
        is_block = False
        for line in file:
            nice_line = line.decode("cp1252")
            if "\n" == nice_line and is_block:
                is_block = False
                block_id += 1

            if is_block:
                data_list = nice_line.split()

                if data_list[0] == "Navn":
                    # Overwrite None with given_name if it exists
                    data["given_name"][-1] = data_list[1]
                else:
                    latitude = float(data_list[0])
                    longitude = float(data_list[1])
                    data["given_name"].append(None)
                    # latitude and longitude is given as decimal minutes
                    data["latitude"].append(latitude / 60)
                    data["longitude"].append(longitude / 60)
                    data["block"].append(block_id)
                    data["timestamp"].append(float(data_list[2]))
                    data["formatted"].append(
                        _format_degree_minutes(latitude, longitude)
                    )

            if "Plottsett" in nice_line:
                is_block = True
    return data


def _make_olex_df(data, only_named=False):
    '''Reads data dictionary and makes DataFrame.
    If only_named is True, then only the named coordinates are returned.'''
    olex_df = pd.DataFrame(data)
    if only_named:
        name_filter = pd.notna(olex_df.given_name)
        return olex_df.loc[name_filter]
    return olex_df


def _get_named_blocks(olex_df):
    '''Get any block with named coordinate.'''
    name_filter = pd.notna(olex_df.given_name)
    named_blocks = olex_df.loc[name_filter, "block"].unique()
    return named_blocks


def _calculate_geodesic(olex_df):
    '''Calculate line length and azimuthal angle from olex_df.
    Return DataFrame with the values and anchor names'''
    named_blocks = _get_named_blocks(olex_df)
    olex_block_df = olex_df.set_index('block')
    points = []
    lines = {'length': [], 'azimuth': [], 'anchor_name': []}
    
    for block_nr in named_blocks:
        df_slice = olex_block_df.loc[block_nr]
        points.clear()
        anker = None
        ramme = None
        if isinstance(df_slice, pd.DataFrame):
            for _, row in df_slice.iterrows():
                is_named = pd.notna(row.given_name)
                lat = row.latitude
                long = row.longitude
                if is_named:
                    given_name = row.given_name
                    anker = (lat, long)
                else:
                    ramme = (lat, long)
                if anker and ramme:
                    geodict = Geodesic.WGS84.Inverse(*ramme, *anker)
                    lengde = geodict['s12']
                    azimuth = (geodict['azi1'] + 360) % 360
                    lines['length'].append(lengde)
                    lines['azimuth'].append(azimuth)
                    lines['anchor_name'].append(given_name)
    return pd.DataFrame(lines)


def plot_map(olex_path):
    '''Plot olex map that shows anchor names, all coordinates, line length,
    and bearing. The functions that calculates bearing assumes that each line
    consists of two points, with anchor point being last. olex_df need only
    contain latitude and longitude.'''
    data = _read_olex_object_export(olex_path)
    olex_df = _make_olex_df(data, only_named=False)
    m = fl.Map(
        location=olex_df.loc[0, ["latitude", "longitude"]].values.tolist(),
        zoom_start=15
    )
    df_blocks = olex_df.set_index("block")
    max_block = olex_df["block"].max()
    named_blocks = _get_named_blocks(olex_df)

    for _, row in olex_df.iterrows():
        fl.Marker([row["latitude"], row["longitude"]],
                  tooltip=row["given_name"],
                  popup=row["formatted"]).add_to(m)

    for block_nr in range(1, max_block+1):
        df_slice = df_blocks.loc[block_nr, ["latitude", "longitude"]]
        s12 = 0
        bearing = None
        for i in range(len(df_slice)-1):
            s12 = Geodesic.WGS84.Inverse(*df_slice.iloc[i+1],
                                         *df_slice.iloc[i])["s12"]
        if block_nr in named_blocks:
            bearing = Geodesic.WGS84.Inverse(*df_slice.iloc[0],
                                             *df_slice.iloc[1])["azi1"]
            bearing = (bearing + 360) % 360

        tooltip = '{:3.1f} m'.format(s12)
        if bearing: 
            tooltip += ', {:3.1f}\N{degree sign}'.format(bearing)

        fl.PolyLine(df_slice.values, tooltip=tooltip).add_to(m)
    return m


def make_buildup_form(olex_path):
    '''Make build up form from olex file.'''
    data = _read_olex_object_export(olex_path)
    olex_full = _make_olex_df(data, only_named=False)
    olex_anchor = _make_olex_df(data, only_named=True).reset_index()
    geodesics = _calculate_geodesic(olex_full)
    olex_geodesic = pd.concat([olex_anchor, geodesics], axis=1)
    assert (olex_geodesic.given_name == olex_geodesic.anchor_name).all(),\
        'Anchor positions has not been inferred correctly.\
        Geodesics may be wrong.'
    cols_of_interest = [
        'given_name',
        'length',
        'azimuth',
        'formatted'
    ]
    pretty_header = [
        'Line',
        'Lengde [m]',
        'Kurs [\N{degree sign}]',
        'GPS'
    ]
    buildup_form = olex_geodesic[cols_of_interest]
    buildup_form.columns = pretty_header
    return buildup_form
