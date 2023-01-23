import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.cluster import DBSCAN


def query_power_plants() -> str:
	"""
	Overpass API query to be run

	:param:
	:return: query string
	:rtype: str

	The query looks for power plants and generators with a numerical value for the output electricity
	inside the area of Austria.
	"""
	query = """
	// define output format
	[out:json];
	
	// gather results
	area[name="Österreich"]->.searchArea;
	(
	  nwr["power"="plant"]["plant:output:electricity"](if:t["plant:output:electricity"]!="yes")(area.searchArea);
	  nwr["power"="generator"]["generator:output:electricity"](if:t["generator:output:electricity"]!="yes")(area.searchArea);
	);
	
	// print results
	out tags center;
	"""
	return query


# def read_csv(csv_file) -> pd.DataFrame:
#     df = pd.read_csv(csv_file, index_col='@id')
#     # Combine plants and generators
#     df['source'] = df['plant:source'].fillna(df['generator:source'])
#     df['capacity'] = df['plant:output:electricity'].fillna(df['generator:output:electricity'])
#     df['method'] = df['plant:method'].fillna(df['generator:method'])
#     # Drop unuseful columns and rename the ones that we will use
#     df.drop(['plant:source', 'generator:source', 'plant:output:electricity', 'generator:output:electricity',
#              'plant:method', 'generator:method'], axis=1, inplace=True)
#     df.rename(columns={'@type':'OSM type', '@lat':'lat', '@lon':'lon'}, inplace=True)
#     df.rename_axis('id', axis='index', inplace=True)
#
#     if df.index.has_duplicates:
#         raise Exception('The index contains duplicates, please change the duplicated id to a valid one')
#
#     return df

def format_power_plants(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Format and clean power plants DataFrame

	:param pd.DataFrame df: power plants DataFrame
	:return: formatted power plants DataFrame
	:rtype: pd.DataFrame
	"""
	# Set id of each element as the index of the DataFrame
	df.set_index(df.id, inplace=True, verify_integrity=True)
	df.index.name = 'id'

	# Create one DataFrame for each "type" value as they need to be handled separately
	df_node = df.loc[df.type == 'node'][['lat', 'lon', 'tags', 'center']]
	df_way = df.loc[df.type == 'way'][['lat', 'lon', 'tags', 'center']]
	df_relation = df.loc[df.type == 'relation'][['lat', 'lon', 'tags', 'center']]

	# Get latitude and longitude of the centroid of power plants with types "way" and "relation"
	df_way['lat'] = [pos['lat'] for pos in df_way.center]
	df_way['lon'] = [pos['lon'] for pos in df_way.center]
	df_relation['lat'] = [pos['lat'] for pos in df_relation.center]
	df_relation['lon'] = [pos['lon'] for pos in df_relation.center]

	# Join the three DataFrames
	df_pp = pd.concat([df_node, df_way, df_relation])

	# Delete power plants operated by ÖBB because they are part of the railway system
	df_pp.drop(df_pp.loc[df_pp.tags.apply(lambda x: 'ÖBB' in x['operator'] if 'operator' in x.keys() else False)].index, inplace=True)

	# Create columns from the information in the "tags" column
	df_pp['OSM type'] = df.type
	df_pp['name'] = df_pp.tags.apply(lambda x: x['name'] if 'name' in x.keys() else np.NaN)
	df_pp['power'] = df_pp.tags.apply(lambda x: x['power'] if 'power' in x.keys() else np.NaN)
	df_pp['generator:source'] = df_pp.tags.apply(lambda x: x['generator:source'] if 'generator:source' in x.keys() else np.NaN)
	df_pp['generator:output:electricity'] = df_pp.tags.apply(lambda x: x['generator:output:electricity'] if 'generator:output:electricity' in x.keys() else np.NaN)
	df_pp['generator:method'] = df_pp.tags.apply(lambda x: x['generator:method'] if 'generator:method' in x.keys() else np.NaN)
	df_pp['plant:source'] = df_pp.tags.apply(lambda x: x['plant:source'] if 'plant:source' in x.keys() else np.NaN)
	df_pp['plant:output:electricity'] = df_pp.tags.apply(lambda x: x['plant:output:electricity'] if 'plant:output:electricity' in x.keys() else np.NaN)
	df_pp['plant:method'] = df_pp.tags.apply(lambda x: x['plant:method'] if 'plant:method' in x.keys() else np.NaN)

	# Combine plants and generators
	df_pp['source'] = df_pp['plant:source'].fillna(df_pp['generator:source'])
	df_pp['capacity'] = df_pp['plant:output:electricity'].fillna(df_pp['generator:output:electricity'])
	df_pp['method'] = df_pp['plant:method'].fillna(df_pp['generator:method'])
	# Drop unuseful columns and rename the ones that we will use
	df_pp.drop(['plant:source', 'generator:source', 'plant:output:electricity', 'generator:output:electricity',
				'plant:method', 'generator:method', 'tags', 'center'], axis=1, inplace=True)

	# Drop entries that don't have a value for the capacity
	df_pp.drop(df_pp.loc[df_pp.capacity.apply(lambda x: any(char.isdigit() for char in x) == False)].index, inplace=True)

	if df_pp.index.has_duplicates:
		raise Exception('The index contains duplicates, please change the duplicated id to a valid one')

	return df_pp


def convert_capacity(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Convert 'capacity' column to values in MW

	:param pd.DataFrame df: power plants DataFrame
	:return: power plants DataFrame with capacity values in MW
	:rtype: pd.DataFrame
	"""
	# Create DataFrame from the values in the "capacity" column
	capacity_df = df['capacity'].str.split(expand=True)
	capacity_df.rename(columns={0: 'value', 1: 'unit'}, inplace=True)

	# Manually change capacity from elements with unit == MWh
	capacity_df.loc[5398756504, ['value', 'unit']] = [1.2, 'MW']
	capacity_df.loc[8339177107, ['value', 'unit']] = [0.82, 'MW']
	capacity_df.loc[8339205885, ['value', 'unit']] = [1.4, 'MW']
	capacity_df.loc[8339205888, ['value', 'unit']] = [1.8, 'MW']
	capacity_df.loc[8339205968, ['value', 'unit']] = [0.6, 'MW']
	capacity_df.loc[8339205997, ['value', 'unit']] = [0.9, 'MW']

	# Drop entries without a capacity value or with a capacity equal to 0 and format values
	problematic_values = capacity_df.loc[pd.to_numeric(capacity_df.value, errors='coerce').isnull()].value.unique().tolist()

	for val in problematic_values:
		problematic_entries = capacity_df.loc[capacity_df.value == val].index

		if any(char.isdigit() is True for char in val):
			capacity_df.loc[problematic_entries, 'value'] = capacity_df.loc[problematic_entries].value.str.replace('""', '')

			if (capacity_df.loc[problematic_entries].unit.isnull()).all():
				for i in range (len(val)):
					if val[i].isalpha():
						break_idx = i
						break
				capacity_df.loc[capacity_df.value == val, 'unit'] = capacity_df.loc[problematic_entries].value.str[break_idx:]
				capacity_df.loc[capacity_df.value == val, 'value'] = capacity_df.loc[problematic_entries].value.str[0:break_idx]
				val = capacity_df.loc[problematic_entries].value.unique()[0]

			if ',' in val:
				capacity_df.loc[capacity_df.value == val, 'value'] = capacity_df.loc[problematic_entries].value.str.replace(',', '.')

		else:
			capacity_df.drop(problematic_entries, inplace=True)
			df.drop(problematic_entries, inplace=True)

	capacity_df.value = pd.to_numeric(capacity_df.value, errors='raise')

	# Delete entries with a capacity of 0 (some plants only provide thermal power and also some entries have missing data)
	zero_capacity_entries = capacity_df.loc[capacity_df.value == 0].index
	capacity_df.drop(zero_capacity_entries, inplace=True)
	df.drop(zero_capacity_entries, inplace=True)

	# Check that all entries have the correct format
	problematic_values = capacity_df.loc[pd.to_numeric(capacity_df.value, errors='coerce').isnull()].value.unique().tolist()
	if problematic_values != []:
		print('There are still bad formatted entries in the dataset:\n', problematic_values)

	# Convert units to MW -> create DataFrame with incorrect units
	kw_to_mw = capacity_df.loc[(capacity_df.unit.str.lower() == 'kw') | (capacity_df.unit.str.lower() == 'kwp') | (capacity_df.unit.str.lower() == 'kva')]
	w_to_mw = capacity_df.loc[(capacity_df.unit.str.lower() == 'w')]
	gw_to_mw = capacity_df.loc[(capacity_df.unit.str.lower() == 'gw')]

	# Drop entries with units in GW (incorrect values)
	capacity_df.drop(gw_to_mw.index, axis=0, inplace=True)
	# Drop entries also from the power plants DataFrame
	df.drop(gw_to_mw.index, axis=0, inplace=True)

	# Convert units
	capacity_df.loc[kw_to_mw.index, 'value'] = capacity_df.value/1000
	capacity_df.loc[w_to_mw.index, 'value'] = capacity_df.value/1e6
	capacity_df.unit = 'MW'

	# Put converted capacity values into the power plants DataFrame
	df['capacity'] = capacity_df.value
	df.rename(columns={'capacity': 'capacity(MW)'}, inplace=True)

	return df

def reassign_source_values(df:pd.DataFrame) -> pd.DataFrame:
	"""
	Modify source for some power plants and create 'template' column

	:param pd.DataFrame df: power plants DataFrame
	:return: power plants DataFrame with new source values
	:rtype: pd.DataFrame

	The 'template' column is used to write the csv file. This column maps each power plant's source to a
	PowerFactory template for that type of power plant.
	"""
	df.loc[df.source == 'biogas', 'source'] = 'biomass' # combine biomass and biogas into biomass
	df.loc[df.source == 'gas;waste;biomass', 'source'] = 'gas' # manually checked
	df.loc[df.source == 'gas;oil;biomass', 'source'] = 'gas' # manually checked
	df.loc[df.source == 'waste;gas', 'source'] = 'waste' # manually checked
	df.loc[df.source == 'hydro;solar', 'source'] = 'hydro' # manually checked

	# Create new column "template" which will be used to create the csv file according to PF format
	d_type = {'wind': 'Wind', 'solar': 'PV', 'hydro': 'Hydro', 'biomass': 'Gas', 'gas': 'Gas', 'gas;waste': 'Gas', 'waste': 'Gas'}
	df['template'] = df.source.map(d_type).tolist()

	hydro_methods = ['run-of-the-river', 'water-storage', 'water-pumped-storage']
	condition = ~df.method.isin(hydro_methods)
	df.loc[df.source == 'hydro', 'source'] = df.loc[df.source == 'hydro'].source.where(condition, df.source + ': ' + df.method)
	df.drop('method', axis=1, inplace=True)

	# Delete power plants with source == "battery" and generators with source == "gas" (they are redundant):
	df.drop(df.loc[df.source == 'battery'].index, inplace=True)
	df.drop(df.loc[(df.power == 'generator') & (df.source.isin(['gas']) == True)].index, inplace=True)
	# Delete coal power plant:
	df.drop(df.loc[df.source == 'coal'].index, inplace=True)

	# Correct units of incoherent data points (solar power plants with infeasible capacity values (>100 MW)) and input missing capacity value:
	df.loc[5199998365, 'capacity(MW)'] = df.loc[5199998365]['capacity(MW)']/1000
	df.loc[5200203619, 'capacity(MW)'] = df.loc[5200203619]['capacity(MW)']/1000

	# Drop duplicated elements (same name and capacity):
	duplicated = df.loc[(df[['name', 'capacity(MW)']].duplicated()) & (df.name.isnull() == False) &
						   (df.source.isin(['wind', 'solar']) == False)]
	df.drop(duplicated.index, inplace=True)

	return df


def print_agg_cap_by_source(df):
	"""
	Show power plant capacity summary with capacities aggregated by power plant source type

	:param pd.DataFrame df: power plants DataFrame
	"""
	df_total_capacity_by_source = df.groupby('template')['capacity(MW)'].agg(['sum'])
	print('Total installed capacity: ', float(df_total_capacity_by_source.sum()), 'MW')
	print('\nAggregated installed capacity by source:\n', df_total_capacity_by_source)

	import matplotlib
	matplotlib.rcParams['mathtext.fontset'] = 'stix'
	matplotlib.rcParams['font.family'] = 'STIXGeneral'
	plt.rc('font', size=14)

	colormap = cm.tab10([7, 0, 1, 2])
	sizes = np.asarray(df_total_capacity_by_source['sum'])/float(df_total_capacity_by_source.sum())*100
	plt.pie(sizes, autopct='%1.1f%%', shadow=True, startangle=0, colors=colormap)
	plt.axis('equal')
	plt.legend(['Gas', 'Hydro', 'PV', 'Wind'])
	plt.show()


def plot_power_plant_map(df):
	"""
	Plot power plants weighted by their capacity on austrian map

	:param pd.DataFrame df: power plants DataFrame
	"""
	import matplotlib
	matplotlib.rcParams['mathtext.fontset'] = 'stix'
	matplotlib.rcParams['font.family'] = 'STIXGeneral'
	plt.rc('font', size=14)

	gdf_powplant = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

	# We restrict to Austria.
	ax = gpd.read_file('../raw_data/maps/austria_bezirke/austria_bezirke.shp').geometry
	ax = ax.to_crs('EPSG:4326')
	ax = ax.plot(color='white', edgecolor='black')

	# We can now plot our GeoDataFrame
	colormap = cm.tab10([0, 7, 1, 2])
	weight = gdf_powplant["capacity(MW)"].astype(float)
	gdf_powplant.plot(ax=ax, x="lon", y="lat", kind="scatter",
					  c=colormap[df.template.factorize()[0]], s=weight, figsize=(17, 17))

	markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colormap]
	plt.legend(markers, df.template.unique(), numpoints=1, loc='upper left')
	plt.show()


def cluster_data(df: pd.DataFrame, dist: int) -> pd.Series:
	"""
	Cluster power plants using DBSCAN
	
	:param pd.DataFrame df: power plants DataFrame
	:param int dist: maximum distance between two samples for one to be considered as in the neighborhood of the other
	:return: clusters resulting from DBSCAN algorithm
	:rtype: pd.Series 
	"""
	coords = df[['lat', 'lon']].to_numpy()
	ids = df.index.to_numpy()
	m_per_radian = 6371.0088*1000
	epsilon = dist / m_per_radian
	db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
	cluster_labels = db.labels_
	num_clusters = len(set(cluster_labels))
	clusters = pd.Series([ids[cluster_labels == n] for n in range(num_clusters)])
	#print('Number of clusters: {}'.format(num_clusters))
	#clusters.to_csv("../data/csv/generated/clusters.csv")

	return clusters


def remove_redundant_generators(clusters: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
	"""
	Remove generators whose capacity is already included in a near power plant of the same source

	:param pd.Series clusters: clusters resulting from :func:`cluster_data`
	:param pd.DataFrame df: power plants DataFrame
	:return: power plants DataFrame without redundant generators
	:rtype: pd.DataFrame
	"""
	for cluster in clusters.values:
		generators = [x for x in cluster if df.power[x] == 'generator']
		plants = [x for x in cluster if df.power[x] == 'plant']
		plant_cap = df.loc[plants]['capacity(MW)'].sum()
		# If plant_cap-10 <= df.loc[generators]['capacity(MW)'].sum() <= plant_cap+10:
		if len(plants) >= 1:
			#print(plants)
			gen_to_drop = [gen for gen in generators if any(df.source[gen] in plant for plant in df.source[plants].to_list())]
			df.drop(gen_to_drop, inplace=True)
	return df


def load_atlas_data(csv_file) -> pd.DataFrame:
	"""
	Get solar generation installations data from https://www.statistik.at/atlas/

	:param str csv_file: path to csv file with new data
	:return: DataFrame with formatted new solar installations data
	:rtype: pd.DataFrame

	The new solar power plants data is based on the installed PV systems supported by the funding initiatives
	`Climate and Energy Fund <https://www.klimafonds.gv.at/>`_ and `OeMAG <https://www.oem-ag.at/de/home/>`_.
	"""

	atlas_df = pd.read_csv(csv_file, skiprows=8, delimiter=';', index_col='ID')
	atlas_df.rename(columns={'Name':'name', 'Wert':'kW per 1000 hab.', 'abs.':'capacity(MW)'}, inplace=True)
	atlas_df.rename_axis('GKZ', axis='index', inplace=True)
	atlas_df.drop('Unnamed: 4', axis=1, inplace=True)

	# Convert "kW" column to float
	atlas_df.loc[atlas_df.index, 'capacity(MW)'] = atlas_df['capacity(MW)'].str.replace(',', '.')
	atlas_df.loc[atlas_df.index, 'capacity(MW)'] = atlas_df['capacity(MW)'].str.replace('-', '0')
	atlas_df.loc[atlas_df.index, 'capacity(MW)'] = atlas_df['capacity(MW)'].apply(lambda x: float(x)/1000)

	# Drop entries with installed capacity equal to 0
	atlas_df.drop(atlas_df.loc[atlas_df['capacity(MW)'] == 0].index, inplace=True)

	if atlas_df.index.has_duplicates:
		raise Exception('The index contains duplicates, please change the duplicated id to a valid one')

	return atlas_df


def load_gps_data(csv_file) -> pd.DataFrame:
	"""
	Get geometric data of the Austrian municipalities from https://data.statistik.gv.at/web/meta.jsp?dataset=OGDEXT_GEM_1

	:param str csv_file: path to csv file with data
	:return: DataFrame with centroid coordinates of each municipality
	:rtype: pd.DataFrame
	"""
	shp_df = gpd.read_file(csv_file)
	shp_df.set_index('id', inplace=True)
	shp_df.index = shp_df.index.astype(int)
	shp_df.rename_axis('GKZ', axis='index', inplace=True)

	shp_df['coords'] = shp_df.geometry.centroid.to_crs('EPSG:4326')
	shp_df['lat'] = shp_df.coords.apply(lambda x: x.y)
	shp_df['lon'] = shp_df.coords.apply(lambda x: x.x)

	if shp_df.index.has_duplicates:
		raise Exception('The index contains duplicates, please change the duplicated ids to a valid one')

	return shp_df


def update_solar(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Update solar power plants with new data

	:param pd.DataFrame df: power plants DataFrame
	:return: new power plants DataFrame with updated data
	:rtype: pd.DataFrame
	"""
	df_atlas = load_atlas_data('../raw_data/Power_Plants/photovoltaikanlagen_in_österreich.csv')
	shp_df = load_gps_data('../raw_data/maps/austria_gemeinde/austria_gemeinde.shp')

	# Merge dataframes and input manually the coordinates for the city of Vienna
	df_atlas = df_atlas.merge(shp_df[['lat', 'lon']], left_index=True, right_index=True, how='left')
	df_atlas.loc[90001, 'lat'] = 48.2083
	df_atlas.loc[90001, 'lon'] = 16.373

	# Put new data into power plants DataFrame with the following format:
	"""
	id: GKZ
	OSM type: "Non-OSM"
	power: "PV systems"
	source: "solar"
	"""
	df_atlas.rename_axis('id', axis='index', inplace=True)
	df_atlas['OSM type'] = 'Non-OSM'
	df_atlas['power'] = 'PV systems'
	df_atlas['source'] = 'solar'
	df_atlas.drop(columns='kW per 1000 hab.', inplace=True)

	return pd.concat([df, df_atlas], verify_integrity=True)
