import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
import geopandas as gpd


def query_terminals():
    """
    Overpass API query to be run

    :param:
    :return: query string
    :rtype: str

    The query looks for substations with a voltage value >= 100kV inside the area of Austria and filters out
    the substations with a frequency of 16.7Hz, which belong to the railway system.
    """
    query = """
    // define output format
    [out:json];

    // gather results
    area[name="Österreich"]->.searchArea;
    (
      nwr["power"="substation"](if:(t["voltage"]>=100000) && (t["frequency"]!=16.7))(area.searchArea);
    );

    // print results
    out body geom;
    """
    return query


def format_substations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format and clean substations DataFrame

    :param pd.DataFrame df: substations DataFrame
    :return: formatted substations DataFrame
    :rtype: pd.DataFrame
    """
    # Set id of each element as the index of the DataFrame
    df.set_index(df.type.str[0]+df.id.astype(str), inplace=True, verify_integrity=True)
    df.index.name = 'id'

    # Drop entries with a voltage value of "medium" (this voltage value is considered invalid)
    df.drop(df.loc[df.tags.apply(lambda x: x['voltage'] == 'medium')].index, inplace=True)

    # Create columns from the information in the "tags" column
    df['name'] = df.tags.apply(lambda x: x['name'] if 'name' in x.keys() else np.NaN)
    df['voltage'] = df.tags.apply(lambda x: x['voltage'] if 'voltage' in x.keys() else np.NaN)
    df['frequency'] = df.tags.apply(lambda x: x['frequency'] if 'frequency' in x.keys() else np.NaN)

    # Create one DataFrame for each "type" value as they need to be handled separately
    df_subs_node = df.loc[df.type == 'node'][['lat', 'lon', 'name', 'voltage', 'frequency']]
    df_subs_way = df.loc[df.type == 'way'][['geometry', 'name', 'voltage', 'frequency']]
    df_subs_relation = df.loc[df.type == 'relation'][['members', 'name', 'voltage', 'frequency']]

    # Get geometry coordinates for substations of "type" == "way"
    df_subs_way.geometry = df_subs_way.geometry.apply(lambda x: [[d['lat'], d['lon']] for d in x])

    def get_relation_geometry(members_list: list) -> list:
        geometry = []
        for member in members_list:
            if member['role'] == 'outer':
                geometry += [[d['lat'], d['lon']] for d in member['geometry']]
        return geometry
    # Get geometry coordinates for substations of "type" == "relation"
    df_subs_relation['geometry'] = df_subs_relation.members.apply(get_relation_geometry)
    # Get geometry coordinates for substations of "type" == "node"
    df_subs_node['geometry'] = df_subs_node.apply(lambda x: [[x['lat'], x['lon']]], axis=1)

    # Drop unused columns
    df_subs_node.drop(columns=['lat', 'lon'], inplace=True)
    df_subs_relation.drop(columns=['members'], inplace=True)

    # Join the three DataFrames
    df_subs = pd.concat([df_subs_node, df_subs_way, df_subs_relation])

    # Delete substations that have a frequency of 16.7 Hz (belong to railway system)
    df_subs.drop(df_subs.loc[(df_subs.frequency.isna() == False) & (df_subs.frequency.str.contains('16.7'))].index, inplace=True)
    # Delete substations that don't have any voltage > 100kV
    df_subs.drop(df_subs.loc[df_subs.voltage.apply(lambda x: all([float(volt) < 100000 for volt in x.split(';')]))].index, inplace=True)

    # Remove duplicated substations (different instances with same geometry)
    df_subs.drop(df_subs.loc[df_subs.geometry.apply(lambda x: tuple(x)).duplicated(keep='first')].index, inplace=True)

    # Duplicate substations with more than one voltage level
    df_subs['voltage'] = df_subs.voltage.str.split(';')
    df_subs = df_subs.explode('voltage')
    df_subs['voltage'] = df_subs['voltage'].astype(float)/1000

    # Take only the substation nodes that have a voltage > 110kV
    df_subs = df_subs.loc[df_subs.voltage >= 110]

    # Delete substations that are redundant because they are already inside another substation (special cases)
    df_subs.drop(df_subs.loc[df_subs.name == 'SF6-Halle 380 kV'].index, inplace=True)

    # Handle nan in "name" -> fill nan values with an empty string
    df_subs.name.fillna(' ', inplace=True)

    # Save the "id" information in a new column and create new index with the needed format for the csv file
    df_subs['id'] = df_subs.index
    new_index = pd.Index(['ElmTerm_{0:04}'.format(i) for i in range(1, len(df_subs)+1)])
    df_subs.set_index(new_index, inplace=True)

    return df_subs


def convert_substation_areas(df_subs: pd.DataFrame) -> pd.DataFrame:
    """
    Transform substations to circular geometries

    :param pd.DataFrame df_subs: substations DataFrame
    :return: substations DataFrame with new geometry
    :rtype: pd.DataFrame

    Uses the Mercator projection to find the Hausdorff distance between the substation's centroid and
    the substation's perimeter. Then creates a circle with a minimum radius of 200 meters and proportional
    to the Hausdorff distance. And finally converts the circle to GPS coordinates and assigns it as the
    substation's geometry.
    """
    # Create DataFrame with substations whose "type" is "way" or "relation" as their geometry is a polygon
    # (node substations' geometry is a point)
    df_subs_areas = df_subs.loc[df_subs.id.str.contains('|'.join(['w', 'r']))].geometry.apply(lambda x: Polygon(x))

    import pyproj

    # Define mercator projection and wgs84 projection
    wgs84 = pyproj.CRS('EPSG:4326')
    mercator = pyproj.CRS('EPSG:3857')
    # Projection from wgs84 to mercator
    project = pyproj.Transformer.from_crs(wgs84, mercator, always_xy=True).transform
    # Projection from mercator to wgs84
    project2 = pyproj.Transformer.from_crs(mercator, wgs84, always_xy=True).transform

    def aux_func(x):
        # Get centroid of the substation's area
        centroid = x.centroid

#         if transform(project, x).area > 50000:
#             radius = 500.0
#         else:
#             radius = 200.0

        # Project the substation's centroid and area to mercator
        point_transformed = transform(project, centroid)
        area_transformed = transform(project, x)
        # Calculate Hausdorff distance between the substation's perimeter and its centroid
        hausdorff_dist = area_transformed.exterior.hausdorff_distance(point_transformed)
        # Define radius for the new circular area
        radius = 1.5*hausdorff_dist if 1.5*hausdorff_dist > 200 else 200
        # Create circulate area
        buffer = point_transformed.buffer(radius)
        # Get the polygon with lat lon coordinates in the wgs84 system
        circle_poly = transform(project2, buffer)

        return list(circle_poly.exterior.coords)

    # Apply function to convert substation areas into circles to each element of the DataFrame
    df_subs_areas_circle = df_subs_areas.apply(aux_func)
    # Assign new geometry to entries in the substations DataFrame
    df_subs.loc[df_subs_areas_circle.index, 'geometry'] = df_subs_areas_circle

    # Add substations to DataFrame where there should be a substation but the data is missing (special cases)
    locations = [Point(46.6433697, 14.9357567), Point(48.4337037, 13.9863834), Point(46.8018598, 12.7493858),
                 Point(47.7509128, 13.0361237), Point(47.64103, 12.20077), Point(47.2987425, 11.5951989)]
    voltages = [110.0, 110.0, 110.0, 110.0, 110.0, 110.0]
    for i in range(len(locations)):
        point_transformed = transform(project, locations[i])
        radius = 200
        buffer = point_transformed.buffer(radius)
        circle_poly = transform(project2, buffer)
        df_subs.loc['ElmTerm_{0:04}'.format(len(df_subs)+1)] = ['manually added', voltages[i], 50, list(circle_poly.exterior.coords), 'wXXXXXX']
    
    return df_subs


def assign_areas_and_zones(df_subs: pd.DataFrame) -> pd.DataFrame:
    """
    Assign areas and zones to terminals using NUTS (Nomenclature of Territorial Units for Statistics)

    :param pd.DataFrame df_subs: terminals DataFrame
    :return: terminals DataFrame with assigned zone and area
    :retype: pd.DataFrame

    NUTS2 regions define the areas and NUTS3 define the zones. Tieline terminals are assigned the code 'TL' for the
    area and the zone fields.
    """
    nuts2 = gpd.read_file('../raw_data/maps/STATISTIK_AUSTRIA_NUTS2_20160101/STATISTIK_AUSTRIA_NUTS2_20160101.shp', encoding='utf-8').to_crs('EPSG:4326')
    nuts3 = gpd.read_file('../raw_data/maps/STATISTIK_AUSTRIA_NUTS3_20220101/STATISTIK_AUSTRIA_NUTS3_20220101.shp', encoding='utf-8').to_crs('EPSG:4326')
#     display(nuts2)
#     display(nuts3)

    subs_1 = df_subs.loc[df_subs.id.str.contains('|'.join(['w', 'r', ' '])) & ~df_subs.id.isna()][['name', 'geometry']]
    subs_2 = df_subs.loc[(df_subs.name != 'tieline terminal') & ((df_subs.id.isna()) | (df_subs.id.str.contains('n')))][['name', 'geometry']]
    subs_1['sh_terminals'] = subs_1.geometry.apply(lambda p: transform(lambda x, y: (y, x), LineString(p).centroid))
    subs_2['sh_terminals'] = subs_2.geometry.apply(lambda p: transform(lambda x, y: (y, x), Point(p[0])))
    terminals = pd.concat([subs_1, subs_2])

    areas = [next(nuts2.ID[i] for i in range(len(nuts2)) if nuts2.geometry[i].intersects(terminal)) for terminal in terminals.sh_terminals]
    zones = [next(nuts3.id[i] for i in range(len(nuts3)) if nuts3.geometry[i].intersects(terminal)) for terminal in terminals.sh_terminals]

    terminals['Area'] = areas
    terminals['Zone'] = zones

    df_subs[['Area', 'Zone']] = 'TL'
    df_subs.loc[terminals.index, ['Area', 'Zone']] = terminals[['Area', 'Zone']]

    return df_subs


def create_trafos_csv(df_subs: pd.DataFrame) -> pd.DataFrame:
    """
    Create transformers data and write it to a csv file

    :param pd.DataFrame df_subs: terminals DataFrame
    :return: transformers DataFrame
    :rtype: pd.DataFrame
    """
    df_trafos = pd.DataFrame(columns=['loc_name','desc','buslv.cterm','buslv.__switch__.on_off','bushv.cterm','bushv.__switch__.on_off', 'typ_id'])

    # Get terminals with duplicated geometry as these are the ones that need a transformer between them
    tr1 = df_subs.loc[~pd.isna(df_subs.id) & df_subs.geometry.duplicated(keep=False)].sort_values(by='geometry')
    tr2 = df_subs.loc[pd.isna(df_subs.id) & df_subs.geometry.duplicated(keep=False)].sort_values(by='geometry')
    trafo = pd.concat([tr1, tr2])

    # Assign a group number to each different terminal geometry
    trafo['geometry'] = trafo.geometry.astype(str)
    trafo['group_num'] = trafo.groupby('geometry').ngroup()
    trafo.index.name = 'terminals'
    trafo.reset_index(inplace=True)

    # Group entries by group number and aggregate the voltages and terminals ids into lists
    trafo.sort_values(by='voltage', inplace=True)
    trafo = trafo.groupby('group_num').agg({'voltage': list, 'terminals': list})

    d_type = {'[110.0, 220.0]': 'TypTr2_0001', '[220.0, 380.0]': 'TypTr2_0002', '[110.0, 380.0]': 'TypTr2_0003'}

    # Terminals with only two voltage levels (1 transformer)
    trafo_2_vlevels = trafo.loc[trafo.voltage.str.len() == 2]
    trafo_2_vlevels[['term1','term2']] = pd.DataFrame(trafo_2_vlevels.terminals.tolist(), index= trafo_2_vlevels.index)
    df_trafos[['buslv.cterm', 'bushv.cterm']] = trafo_2_vlevels[['term1','term2']]
    df_trafos['typ_id'] = trafo_2_vlevels.voltage.astype(str).map(d_type).tolist()

    # Terminals with three voltage levels (2 (or 3?) transformers)
    trafo_3_vlevels = trafo.loc[trafo.voltage.str.len() == 3]
    trafo_3_vlevels[['term1', 'term2', 'term3']] = pd.DataFrame(trafo_3_vlevels.terminals.tolist(), index= trafo_3_vlevels.index)
    trafo_3_vlevels_terms = pd.concat([pd.DataFrame(trafo_3_vlevels[['term1', 'term2']].rename(columns={'term1':'buslv.cterm', 'term2':'bushv.cterm'})),
                                   pd.DataFrame(trafo_3_vlevels[['term2', 'term3']].rename(columns={'term2':'buslv.cterm', 'term3':'bushv.cterm'}))])
    trafo_3_vlevels_terms['typ_id'] = pd.concat([trafo_3_vlevels.voltage.str[0:2], trafo_3_vlevels.voltage.str[1:]]).astype(str).map(d_type).tolist()
    df_trafos = pd.concat([df_trafos, trafo_3_vlevels_terms])

    # Finish formatting the DataFrame to create the csv file
    df_trafos[['buslv.__switch__.on_off', 'bushv.__switch__.on_off']] = 1, 1
    df_trafos['desc'] = ' '
    df_trafos['loc_name'] = ['ElmTr2_{0:04}'.format(i) for i in range(1,len(df_trafos)+1)]

    # Write data to csv file
    df_trafos.to_csv('../austrian_grid/csv/Transformers.csv', index=False)

    return df_trafos