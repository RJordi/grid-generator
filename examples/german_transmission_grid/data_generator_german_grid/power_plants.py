import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def query_power_plants(query_area_tag, date) -> str:
	"""
	Overpass API query to be run

	:param:
	:return: query string
	:rtype: str

	The query looks for power plants and generators with a numerical value for the output electricity
	inside the area of Austria.
	"""
	query = f"""
	// define output format
	[out:json]{date}[timeout:1000];
	
	// gather results
	area[{query_area_tag}]->.searchArea;
	(
	nwr["power"="plant"]["plant:output:electricity"](if:t["plant:output:electricity"]!="yes")(area.searchArea);
	nwr["power"="generator"]["generator:output:electricity"](if:t["generator:output:electricity"]!="yes")(area.searchArea);
	);
	
	// print results
	out tags center;
	"""
	return query


def format_power_plants(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Format and clean substations DataFrame

	:param pd.DataFrame df: power plants DataFrame
	:return: formatted power plants DataFrame
	:rtype: pd.DataFrame
	"""
	# Drop duplicated entries
	df.drop_duplicates('id', inplace=True, ignore_index=True)

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

	# Set source value as 'unknown' for NaN cases
	df_pp.loc[df_pp.source.isna(), 'source'] = 'unknown'

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
	capacity_df = df['capacity'].str.split(expand=True, n=1)
	capacity_df.rename(columns={0: 'value', 1: 'unit'}, inplace=True)

	# Drop entries without a capacity value or with a capacity equal to 0 and format values
	problematic_values = capacity_df.loc[pd.to_numeric(capacity_df.value, errors='coerce').isnull()].value.unique().tolist()

	for val in problematic_values:
		problematic_entries = capacity_df.loc[capacity_df.value == val].index

		if any(char.isdigit() is True for char in val):
			capacity_df.loc[problematic_entries, 'value'] = capacity_df.loc[problematic_entries].value.str.replace('""', '')

			if (capacity_df.loc[problematic_entries].unit.isnull()).all():
				for i in range(len(val)):
					if val[i].isalpha():
						break_idx = i
						break
				capacity_df.loc[capacity_df.value == val, 'unit'] = capacity_df.loc[problematic_entries].value.str[break_idx:]
				capacity_df.loc[capacity_df.value == val, 'value'] = capacity_df.loc[problematic_entries].value.str[0:break_idx]
				val = capacity_df.loc[problematic_entries].value.unique()[0]

			if ',' in val:
				capacity_df.loc[capacity_df.value == val, 'value'] = capacity_df.loc[problematic_entries].value.str.replace(',', '.')
			if val[-1].isdigit() is False:
				capacity_df.loc[capacity_df.value == val, 'value'] = val[0:-1]

		else:
			capacity_df.drop(problematic_entries, inplace=True)
			df.drop(problematic_entries, inplace=True)

	capacity_df.value = pd.to_numeric(capacity_df.value, errors='coerce')

	# Delete entries with a capacity of 0 (some plants only provide thermal power and also some entries have missing data)
	zero_capacity_entries = capacity_df.loc[capacity_df.value == 0].index
	capacity_df.drop(zero_capacity_entries, inplace=True)
	df.drop(zero_capacity_entries, inplace=True)

	# Delete entries with a null capacity
	null_capacity_entries = capacity_df.loc[capacity_df.value.isnull()].index
	capacity_df.drop(null_capacity_entries, inplace=True)
	df.drop(null_capacity_entries, inplace=True)

	# Check that all entries have the correct format
	problematic_values = capacity_df.loc[pd.to_numeric(capacity_df.value, errors='coerce').isnull()].value.unique().tolist() #change to 'ignore' to see problematic values
	if problematic_values != []:
		print('There are still bad formatted entries in the dataset:\n', problematic_values)
		capacity_df.loc[capacity_df.value.isin(problematic_values), 'value'] = np.nan

	# Convert units to MW -> create DataFrame with incorrect units
	kw_to_mw = capacity_df.loc[(capacity_df.unit.str.lower() == 'kw') | (capacity_df.unit.str.lower() == 'kwp') | (capacity_df.unit.str.lower() == 'kva')]
	w_to_mw = capacity_df.loc[(capacity_df.unit.str.lower() == 'w')]
	gw_to_mw = capacity_df.loc[(capacity_df.unit.str.lower() == 'gw')]

	# Convert units
	capacity_df.loc[kw_to_mw.index, 'value'] = capacity_df.value/1000
	capacity_df.loc[w_to_mw.index, 'value'] = capacity_df.value/1e6
	capacity_df.loc[gw_to_mw.index, 'value'] = capacity_df.value*1000
	capacity_df.unit = 'MW'

	# Put converted capacity values into the power plants DataFrame
	df['capacity'] = capacity_df.value
	df.rename(columns={'capacity': 'capacity(MW)'}, inplace=True)

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
	df_total_capacity_by_source = df.groupby('source')['capacity(MW)'].agg(['sum'])
	print('Total installed capacity: ', float(df_total_capacity_by_source.sum()), 'MW')
	print('\nAggregated installed capacity by source:\n', df_total_capacity_by_source)


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
