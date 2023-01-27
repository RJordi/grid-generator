import pandas as pd
import numpy as np
import requests
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
import geopandas as gpd
from haversine import haversine_vector, Unit

import power_plants
import lines
import terminals
import tests
# import validate

query_area_tag = 'name="Deutschland"'  # Any other defining tag can be used: https://nominatim.openstreetmap.org/ui/search.html
query_date = '[date:"2016-07-18T00:00:00Z"]'  # To get most recent data pass an empty string
read_data_dir = '../raw_data/'
write_data_dir = '../german_grid/'


def query_overpass(query: str) -> pd.DataFrame:
	"""
	Query Overpass API

	:param str query: query
	:return: response with relevant information
	:rtype: pd.DataFrame
	"""
	OVERPASS_URL = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
	response = requests.get(OVERPASS_URL, params={'data': query})
	if response.ok:
		r = response.json()
		df = pd.DataFrame(r['elements'])
		return df
	else:
		raise ValueError('Could not query data from Overpass API. Given response is: ', response)


def delete_lines_in_subs(df_lines: pd.DataFrame, df_subs: pd.DataFrame) -> pd.DataFrame:
	"""
	Delete lines that start and end inside the same substation

	:param pd.DataFrame df_lines: lines DataFrame
	:param pd.DataFrame df_subs: substations DataFrame
	:return: lines DataFrame with some lines deleted
	:rtype: pd.DataFrame
	"""
	def end_nodes_in_sub(coords_list: list, df_subs: pd.DataFrame) -> bool:
		# Get only substations which have a defined area (avoid using substations which are nodes)
		sub_areas = df_subs.geometry.values.tolist()
		# Get starting and ending points of the line element
		start_point_line, end_point_line = Point(coords_list[0]), Point(coords_list[-1])
		# Iterate through all substations and return True if the starting and the ending point are inside it
		for area in sub_areas:
			if start_point_line.intersects(Polygon(area)) and end_point_line.intersects(Polygon(area)):
				return True
		return False

	# Get lines which start and end in a substation and drop them from the lines DataFrame
	lines_in_subs = df_lines.loc[df_lines.geometry.apply(lambda x: end_nodes_in_sub(x, df_subs))]
	df_lines.drop(lines_in_subs.index, inplace=True)

	return df_lines


def assign_substation_to_lines(df_lines: pd.DataFrame, df_subs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Assign terminal to line end if this one is geometrically inside the terminal

	:param pd.DataFrame df_lines: lines DataFrame
	:param pd.DataFrame df_subs: substations DataFrame
	:return: lines DataFrame with assigned terminals
	:rtype: pd.DataFrame
	:return: substations DataFrame with corresponding terminals
	:rtype: pd.DataFrame

	TODO: Detailed explanation
	"""
	lines_geometry = df_lines.geometry.to_numpy()
	lines_voltage = df_lines.voltage.to_numpy()
	terminals_index = df_subs.index.to_numpy()
	terminals_geometry = df_subs.geometry.to_numpy()
	terminals_voltage = df_subs.voltage.to_numpy()

	new_terminals_index = []
	new_terminals_voltage = []
	new_terminals_geometry = []

	def assign_substation(lines_geometry, lines_voltage, terminals_index, terminals_geometry, terminals_voltage):
		#lines_substations = np.empty(shape=(len(lines_geometry),2), dtype=object).tolist()
		lines_substations = np.empty(shape=(len(lines_geometry),2), dtype=object)
		for i_l in range(len(lines_geometry)):
			assigned_subs = np.array([np.nan, np.nan], dtype=object)
			#assigned_subs = [np.nan, np.nan]
			for e in range(0, -2, -1):
				point = Point(lines_geometry[i_l][e])
				for i_t in range(len(terminals_geometry)):
					if len(terminals_geometry[i_t]) == 1:  # Case where the substation is only one node
						if point == Point(terminals_geometry[i_t][0]):
							assigned_subs[e] = terminals_index[i_t]
							break
					else:  # Case where the substation is a way or a relation
						if point.intersects(Polygon(terminals_geometry[i_t])):
							idxs = [i for i in range(len(terminals_geometry)) if terminals_geometry[i_t]==terminals_geometry[i]]
							if lines_voltage[i_l] in [terminals_voltage[idx] for idx in idxs]:
								idx = next(idx for idx in idxs if lines_voltage[i_l] == terminals_voltage[idx])
								assigned_subs[e] = terminals_index[idx]
								break
							else:
								if terminals_geometry[i_t] in new_terminals_geometry:
									idxs = [i for i in range(len(new_terminals_geometry)) if terminals_geometry[i_t]==new_terminals_geometry[i]]
									if any([new_terminals_voltage[idx] == lines_voltage[i_l] for idx in idxs]):
										idx = next(idx for idx in idxs if new_terminals_voltage[idx] == lines_voltage[i_l])
										assigned_subs[e] = new_terminals_index[idx]
										break
									else:
										new_term_idx = 'ElmTerm_{0:04}'.format(len(terminals_index)+1+len(new_terminals_index))
										new_terminals_index.append(new_term_idx)
										new_terminals_voltage.append(lines_voltage[i_l])
										new_terminals_geometry.append(terminals_geometry[i_t])
										assigned_subs[e] = new_term_idx
										break
								else:
									new_term_idx = 'ElmTerm_{0:04}'.format(len(terminals_index)+1+len(new_terminals_index))
									new_terminals_index.append(new_term_idx)
									new_terminals_voltage.append(lines_voltage[i_l])
									new_terminals_geometry.append(terminals_geometry[i_t])
									assigned_subs[e] = new_term_idx
									break

			lines_substations[i_l] = assigned_subs
		return lines_substations

	df_lines['substations'] = assign_substation(lines_geometry, lines_voltage, terminals_index, terminals_geometry, terminals_voltage).tolist()

	# Append new terminals to the substations DataFrame
	df_new_terminals = pd.DataFrame({'name':'added terminal voltage','voltage':new_terminals_voltage,
									 'geometry':new_terminals_geometry, 'id':' '}, index=new_terminals_index)
	df_subs = pd.concat([df_subs, df_new_terminals])

	# Leave out all substations that are not connected to any line
	used_subs = df_lines.substations.explode().dropna().unique()
	df_subs = df_subs.loc[used_subs]

	return df_lines, df_subs


def assign_substation_to_power_plants(df_pp: pd.DataFrame, df_subs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Assign the closest terminal to each power plant

	:param pd.DataFrame df_pp: power plants DataFrame
	:param pd.DataFrame df_subs: terminals DataFrame
	:return:
		* **df_pp** (*pd.DataFrame*) - power plants DataFrame with assigned terminals
		* **df_pp_agg** (*pd.DataFrame*) - power plants DataFrame with assigned terminals and aggregated capacity

	TODO: Detailed explanation
	"""
	u_terminals = df_subs.loc[~df_subs.name.isin(['virtual terminal', 'tieline terminal'])]
	point_terminals = u_terminals.loc[u_terminals.geometry.str.len() == 1].geometry.apply(lambda x: Point(x[0]).coords[0])
	polygon_terminals = u_terminals.loc[u_terminals.geometry.str.len() > 1].geometry.apply(lambda x: Polygon(x).centroid.coords[0])
	u_terminals_centroids = pd.concat([point_terminals, polygon_terminals])
	pp_centroids = [(lat, lon) for lat, lon in zip(df_pp.lat, df_pp.lon)]
	# nt stands for "nearest terminal"
	nt_idx = np.argmin(haversine_vector(pp_centroids, u_terminals_centroids.tolist(), unit=Unit.KILOMETERS, comb=True), axis=0)
	nt_dist = np.amin(haversine_vector(pp_centroids, u_terminals_centroids.tolist(), unit=Unit.KILOMETERS, comb=True), axis=0)

	pp_assigned_terminals = [u_terminals_centroids.index[i] for i in nt_idx]
	pp_dist_to_terminal = [dist for dist in nt_dist]

	df_pp['assigned_term'] = pp_assigned_terminals
	df_pp['dist_to_term'] = pp_dist_to_terminal

	# Aggregate (sum) capacities for all power plants that have a capacity <= 10MW (considered distributed generation)
	df_pp_distrib = df_pp.loc[df_pp['capacity(MW)'] <= 10]
	grouped_df = df_pp_distrib.groupby(['source', 'assigned_term'], as_index=False).agg({'name': lambda x:'aggregated_pp',
																							'lat': lambda x: 'agg',
																							'lon': lambda x: 'agg',
																							'OSM type': lambda x:'agg',
																							'power': lambda x:'agg', 'capacity(MW)': 'sum',
																							'dist_to_term': lambda x:'agg', 'source': lambda x: list(x)[0]})

	df_pp_agg = df_pp.drop(df_pp_distrib.duplicated(subset=['source', 'assigned_term'], keep=False).index, inplace=False)
	df_pp_agg = pd.concat([df_pp_agg, grouped_df])

	return df_pp, df_pp_agg


def create_virtual_terminals(df_lines: pd.DataFrame, df_subs:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Create virtual terminals at points where more than 2 lines meet

	:param pd.DataFrame df_lines: lines DataFrame
	:param pd.DataFrame df_subs: terminals DataFrame
	:return: lines DataFrame with assigned virtual terminals
	:rtype: pd.DataFrame
	:return: terminals DataFrame with new virtual terminals appended
	:rtype: pd.DataFrame

	TODO: Detailed explanation
	"""
	line_idx = df_lines.index
	conn_seg = df_lines.connecting_segments.values
	subs = df_lines.substations.values
	lines_w_vt = [line_idx[i] for i in range(0, len(line_idx)) if any([len(conn_seg[i][n])>=2 and pd.isna(subs[i][n])==True for n in [0,1]])]
	df1 = df_lines.loc[lines_w_vt] #lines that need a virtual terminal
	conn_seg_df1 = df1.connecting_segments.values
	subs_df1 = df1.substations.values
	df1['fake_term_idx'] = [[-n for n in [0,1] if (len(conn_seg_df1[i][n])>=2 and pd.isna(subs_df1[i][n])==True)] for i in range(len(df1))]
	#df1['fake_term_idx'] = df1.connecting_segments.apply(lambda x: [-x.index(segment) for segment in x if len(segment)>=2])

	df_helper = df1[['voltage', 'circuits', 'ref', 'fake_term_idx']]
	df_helper['lat_start'], df_helper['lon_start'] = df1.geometry.apply(lambda x: x[0][0]), df1.geometry.apply(lambda x: x[0][1])
	df_helper['lat_end'], df_helper['lon_end'] = df1.geometry.apply(lambda x: x[-1][0]), df1.geometry.apply(lambda x: x[-1][1])
	df_helper['fake_term_start'] = df1.apply(lambda row: row.connecting_segments[0] if 0 in row.fake_term_idx else [], axis=1)
	df_helper['fake_term_end'] = df1.apply(lambda row: row.connecting_segments[-1] if -1 in row.fake_term_idx else [], axis=1)

	geometry = df1.apply(lambda row: [row.geometry[idx] for idx in row.fake_term_idx], axis=1).explode()
	df_fake_terms = pd.DataFrame(geometry.apply(lambda x: [x]), columns=['geometry'])
	df_fake_terms['voltage'] = df1.voltage.loc[df_fake_terms.index]
	df_fake_terms.reset_index(inplace=True, drop=True)
	df_fake_terms = df_fake_terms.loc[df_fake_terms.astype(str).drop_duplicates().index]
	df_fake_terms['name'] = 'virtual terminal'
	new_index = pd.Index(['ElmTerm_virtual_{0:04}'.format(i) for i in range(1,len(df_fake_terms)+1)])
	df_fake_terms.set_index(new_index, inplace=True)

	helper_geom = [[[x,y], [k,j]] for x,y,k,j in zip(df_helper.lat_start, df_helper.lon_start, df_helper.lat_end, df_helper.lon_end)]
	df_helper['fake_term_id'] = [[idx for idx in df_fake_terms.index if (df_fake_terms.geometry.loc[idx][0] in helper_geom[i] and df_fake_terms.voltage.loc[idx]==df_helper.voltage[i])] for i in range(len(helper_geom))]

	# Add virtual terminals to the corresponding entries in the lines DataFrame
	new_series_list = []
	for idx, virt_term, way_subs in zip(df_helper.fake_term_idx, df_helper.fake_term_id, df_lines.loc[df_helper.index].substations):
		for e in idx:
			if pd.isna(way_subs[e]):
				way_subs[e] = virt_term[e]
		new_series_list.append(way_subs)
	df_lines.loc[df_helper.index].substations = pd.Series(new_series_list).reindex(df_lines.loc[df_helper.index].index)

	# Add virtual terminals to the substations DataFrame
	df_subs = pd.concat([df_subs, df_fake_terms])

	return df_lines, df_subs


def handle_floating_ends(df_lines: pd.DataFrame, df_subs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Assign a terminal to line ends which are still not connected anywhere

	:param pd.DataFrame df_lines: lines DataFrame
	:param pd.DataFrame df_subs: terminals DataFrame
	:returns:
		* **df_lines** (*pd.DataFrame*) - lines DataFrame with assigned floating end terminals
		* **df_subs** (*pd.DataFrame*) - terminals DataFrame with new floating end terminals appended

	Get lines with at least a floating end. If they have a substation nearby (less than 1 km), assign substation to line and
	add the length value corresponding to the missing segment. Otherwise, add a new terminal to the floating end.
	If the line end is outside the region of Austria, the line is a tieline and a substation is added at the floating end.
	"""

	floating_lines = [any([x[i]==[] and pd.isna(y[i]) for i in range(0,2)]) for x,y in zip(df_lines.connecting_segments, df_lines.substations)]
	floating_lines_df = df_lines.loc[floating_lines]
	non_virtual_subs = df_subs.loc[(df_subs.name != 'virtual terminal') & (df_subs.id.str.contains('n') == False)]
	non_virtual_subs['centroid'] = non_virtual_subs.geometry.apply(lambda x: tuple(Polygon(x).centroid.coords)[0])
	subs_coords = non_virtual_subs.centroid.tolist()

	area_polygon = gpd.read_file(read_data_dir + 'area_geometry.geojson')['geometry'].values[0]  # Exported from OSM as geojson
	# Example query to get the area_geometry.geojson file (get the relation id from https://nominatim.openstreetmap.org/ui/search.html):
	# 	// define output format
	# 	[out:json];
	# 	// gather results
	# 	relation(109166);
	# 	// print results
	# 	out body geom;
	area_polygon = transform(lambda x, y: (y, x), area_polygon)
	ser_list_tielines, ser_list_new_terms = [], []
	max_elm_term_idx = max([int(idx.split('_')[1]) for idx in df_subs.index if len(idx.split('_')) == 2])

	def handle_row(row):
		for i in [0, -1]:
			if row.connecting_segments[i] == [] and pd.isna(row.substations[i]):
				floating_end_coords = row.geometry[i]
				if Point(floating_end_coords).intersects(area_polygon) == False:  # handle tielines
					ser_list_tielines.append(['tieline terminal', row.voltage, np.nan, [floating_end_coords], np.nan])
					row.substations[i] = 'ElmTerm_tieline_{0:04}'.format(len(ser_list_tielines))
				else:
					dist_vector = haversine_vector(floating_end_coords, subs_coords, unit=Unit.KILOMETERS, comb=True)
					ns_idx = np.where(dist_vector == dist_vector.min())[0]
					ns_dist = np.amin(dist_vector)
					if ns_dist < 1:  # assign the closest terminal if there is a terminal less that 1km from the end of the line
						closest_term = non_virtual_subs.index[ns_idx]
						if any([term in row.substations for term in closest_term]) == False:
							closest_term_voltages = non_virtual_subs.voltage.loc[closest_term]
							if closest_term_voltages.isin([row.voltage]).any():
								correct_closest_term = closest_term_voltages.loc[closest_term_voltages == row.voltage].index[0]
								correct_ns_idx = ns_idx[closest_term.tolist().index(correct_closest_term)]
								row.substations[i] = correct_closest_term
								row.length += ns_dist
								row.geometry.append(list(subs_coords[correct_ns_idx]))
							else:  # assign new terminal at the end point
								ser_list_new_terms.append(['floating end terminal', row.voltage, np.nan, [floating_end_coords], np.nan])
								row.substations[i] = 'ElmTerm_{0:04}'.format(len(ser_list_new_terms) + max_elm_term_idx)
						else:  # drop lines if their closest terminal is a repeated one and is less than 1km away
							row.id = 'drop_this_line'
					else:  # assign new terminal at the end point
						ser_list_new_terms.append(['floating end terminal', row.voltage, np.nan, [floating_end_coords], np.nan])
						row.substations[i] = 'ElmTerm_{0:04}'.format(len(ser_list_new_terms) + max_elm_term_idx)
		return row

	floating_lines_df = floating_lines_df.apply(handle_row, axis=1)
	df_lines.drop(floating_lines_df.loc[floating_lines_df.id == 'drop_this_line'].index, inplace=True)
	floating_lines_df.drop(floating_lines_df.loc[floating_lines_df.id == 'drop_this_line'].index, inplace=True)
	df_lines.loc[floating_lines_df.index] = floating_lines_df

	tielines_subs_df = pd.DataFrame(ser_list_tielines, columns=['name', 'voltage', 'frequency', 'geometry', 'id'])
	new_index_tielines_subs_df = pd.Index(['ElmTerm_tieline_{0:04}'.format(i) for i in range(1,len(tielines_subs_df)+1)])
	tielines_subs_df.set_index(new_index_tielines_subs_df, inplace=True)

	new_terms_df = pd.DataFrame(ser_list_new_terms, columns=['name', 'voltage', 'frequency', 'geometry', 'id'])
	new_index_new_terms_df = pd.Index(['ElmTerm_{0:04}'.format(i + max_elm_term_idx) for i in range(1,len(new_terms_df)+1)])
	new_terms_df.set_index(new_index_new_terms_df, inplace=True)
	df_subs = pd.concat([df_subs, tielines_subs_df, new_terms_df])

	return df_lines, df_subs


def handle_special_cases(df_lines, df_subs, df_trafos) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Handle observed troublesome cases "manually"

	:param pd.DataFrame df_lines: lines DataFrame
	:param pd.DataFrame df_subs: temrinals DataFrame
	:param pd.DataFrame df_trafos: transformers DataFrame
	:return:
		* **df_lines** (*pd.DataFrame*) - lines DataFrame with resolved troublesome cases
		* **df_subs** (*pd.DataFrame*) - terminals DataFrame with resolved troublesome cases

	TODO: Detailed explanation
	"""
	# Drop useless bays and busbars
	special_cases = df_lines.loc[(df_lines.line == 'bay') | (df_lines.line=='busbar')]
	ids = special_cases.id.tolist()

	to_drop = special_cases.loc[[all([elem in ids for i in [0,1] for elem in seg[i]]) for seg in special_cases.connecting_segments]]
	df_lines.drop(to_drop.index, inplace=True)

	# Drop lines that can't be connected to any terminal or line (considered as wrong data from OSM)
	exploded_terms = df_lines.substations.explode()
	floating_terminals = df_subs.loc[df_subs.name.str.contains('floating')].index.tolist()
	non_floating_exploded_terms = exploded_terms.loc[~exploded_terms.isin(floating_terminals)]
	one_connection_terminals = (non_floating_exploded_terms.value_counts() == 1)
	one_connection_terminals_list = one_connection_terminals.loc[one_connection_terminals == True].index.tolist()

	problematic_terminals_list = one_connection_terminals_list + floating_terminals
	terminals_with_trafo = pd.concat([df_trafos['buslv.cterm'], df_trafos['bushv.cterm']]).unique().tolist()
	problematic_terminals_list_2 = [elem for elem in problematic_terminals_list if elem not in terminals_with_trafo]

	for index, terminals in df_lines.substations.items():
		if terminals[0] in problematic_terminals_list_2 and terminals[1] in problematic_terminals_list_2:
			# print(index)
			df_lines.drop(index, inplace=True)

	# Drop lines that start and end at the same substation
	df_lines.drop(df_lines.loc[[term[0] == term[1] for term in df_lines.substations]].index, inplace=True)
	# Drop terminals that don't have any line connected to them
	used_subs = df_lines.substations.explode().dropna().unique()
	df_subs = df_subs.loc[used_subs]

	# Drop repeated lines
	df_lines.drop(df_lines.loc[pd.Series([geom+[volt] for geom,volt in zip(df_lines.geometry, df_lines.voltage)]).duplicated(keep=False).values].index, inplace=True)

	return df_lines, df_subs


def create_pf_csv(df_lines: pd.DataFrame, df_subs: pd.DataFrame, df_pp_agg: pd.DataFrame, df_trafos) -> int:
	"""
	Create csv files to generate the PowerFactory model

	:param pd.DataFrame df_lines: lines DataFrame
	:param pd.DataFrame df_subs: terminals DataFrame
	:param pd.DataFrame df_pp_agg: power plants DataFrame
	:param pd.DataFrame df_trafos: transformers DataFrame
	:return: 0
	:rtype: int
	"""
	# Create empty DataFrames
	lines_df_pf = pd.DataFrame(columns=['id','bus1.cterm','bus2.cterm','dline','typ_id','nlnum','desc','GPScoords'])
	substations_df_pf = pd.DataFrame(columns=['id', 'unom', 'desc'])
	pp_df_pf = pd.DataFrame(columns=['id', 'source', 'cterm', 'sgn'])

	# DataFrame for terminals
	substations_df_pf['id'] = df_subs.index
	substations_df_pf[['unom', 'desc']] = df_subs[['voltage', 'name']].set_index(substations_df_pf.index)

	# DataFrame for lines
	lines_df_pf['id'] = ['ElmLne_{0:04}'.format(i) for i in range(1,len(df_lines)+1)]
	d_type = {110.0: 'TypLne_1', 220.0: 'TypLne_2', 380.0: 'TypLne_3'}
	lines_df_pf['typ_id'] = df_lines.voltage.map(d_type).tolist()
	lines_df_pf[['bus1.cterm','bus2.cterm']] = pd.DataFrame(df_lines.substations.tolist())
	df_lines.circuits = df_lines.circuits.astype(int)
	lines_df_pf[['dline','desc', 'GPScoords', 'nlnum']] = df_lines[['length', 'name', 'geometry', 'circuits']].set_index(lines_df_pf.index)

	# DataFrame for power plants
	pp_df_pf['id'] = ['SM_{0:04}'.format(i) for i in range(1, len(df_pp_agg)+1)]
	pp_df_pf[['source', 'cterm']] = df_pp_agg[['source', 'assigned_term']].set_index(pp_df_pf.index)
	pp_df_pf['sgn'] = [df_pp_agg['capacity(MW)'].values[i]*0.9 if df_pp_agg.source.values[i] not in ['PV', 'Wind'] else df_pp_agg['capacity(MW)'].values[i] for i in range(len(df_pp_agg))]

	# Write csv files
	substations_df_pf.to_csv(write_data_dir + 'csv/Terminals.csv', index=False)
	lines_df_pf.to_csv(write_data_dir + 'csv/Lines.csv', index=False)
	pp_df_pf.to_csv(write_data_dir + 'csv/PowerPlants.csv', index=False)
	df_trafos.to_csv(write_data_dir + 'csv/Transformers.csv', index=False)

	return 0


def qgis(df_lines:pd.DataFrame, df_subs:pd.DataFrame, df_trafos:pd.DataFrame, df2:pd.DataFrame, df_pp:pd.DataFrame,
		 df_pp_agg:pd.DataFrame) -> int:
	"""
	Create csv files to visualize data using QGIS desktop application

	:param pd.DataFrame df_lines: lines DataFrame
	:param pd.DataFrame df_subs: terminals DataFrame
	:param pd.DataFrame df_trafos: transformers plants DataFrame
	:param pd.DataFrame df2: substations DataFrame with original geometries (before converting to circles)
	:param pd.DataFrame df_pp: power plants DataFrame
	:param pd.DataFrame df_pp_agg: power plants DataFrame with aggregated distributed generation
	:return: 0
	:rtype: int

	Create csv files with attributes and WKT geometries to visualize the data of the model in QGIS.
	This way layers can be added to the QGIS project using the "Add Delimited Text Layer..." operation.
	The terminals had to be separated into two different files because QGIS can only read data with the same WKT type
	for each file.
	For the power plants also two files are created: one contains the power plants at the original location and before
	being aggregated and the other contains the aggregated power plants located at the position where their assigned
	terminal is positioned.
	"""
	df_qgis_lines = df_lines[['voltage', 'length', 'id', 'substations']]
	df_qgis_lines['sh_lines'] = df_lines.geometry.apply(lambda p: transform(lambda x, y: (y, x), LineString(p)))
	df_qgis_lines['idx'] = df_lines.index

	subs_1 = df_subs.loc[df_subs.id.str.contains('|'.join(['w', 'r', ' '])) & ~df_subs.id.isna()]
	subs_2 = df_subs.loc[(df_subs.id.isna()) | (df_subs.id.str.contains('n'))]
	df_qgis_terminals = subs_1[['voltage', 'name']]
	df_qgis_terminals_nodes = subs_2[['voltage', 'name']]
	df_qgis_terminals['idx'] = subs_1.index
	df_qgis_terminals_nodes['idx'] = subs_2.index
	df_qgis_terminals['sh_terminals'] = subs_1.geometry.apply(lambda p: transform(lambda x, y: (y, x), LineString(p)))
	df_qgis_terminals_nodes['sh_terminals'] = subs_2.geometry.apply(lambda p: transform(lambda x, y: (y, x), Point(p[0])))

	l1 = [[df_trafos.id.tolist()[i] for i in range(len(df_trafos)) if \
	 term == df_trafos['buslv.cterm'].tolist()[i] or term == df_trafos['bushv.cterm'].tolist()[i]] for \
	 term in df_qgis_terminals.index]
	l2 = [[df_trafos.id.tolist()[i] for i in range(len(df_trafos)) if \
	 term == df_trafos['buslv.cterm'].tolist()[i] or term == df_trafos['bushv.cterm'].tolist()[i]] for \
	 term in df_qgis_terminals_nodes.index]

	df_qgis_terminals['trafos'] = l1
	df_qgis_terminals['trafos'].astype(str)
	df_qgis_terminals_nodes['trafos'] = l2
	df_qgis_terminals_nodes['trafos'].astype(str)

	df_subs_og = terminals.format_substations(df2)
	df_qgis_subs_og = df_subs_og.loc[df_subs_og.id.str.contains('|'.join(['w', 'r'])) & ~df_subs_og.id.isna()][['name', 'geometry']]
	df_qgis_subs_og['geometry'] = df_qgis_subs_og.geometry.apply(lambda p: transform(lambda x, y: (y, x), Polygon(p)))

	df_qgis_pp_og = df_pp[['name', 'source', 'capacity(MW)', 'assigned_term', 'dist_to_term']]
	df_qgis_pp_og['sh_pp_og'] = [Point(lon, lat) for lon,lat in zip(df_pp.lon, df_pp.lat)] # Raises a deprecation warning but it's fine to ignore it (https://shapely.readthedocs.io/en/stable/migration.html#creating-numpy-arrays-of-geometry-objects)
	df_qgis_pp_og['idx'] = ['SM_{0:04}'.format(i) for i in range(1,len(df_pp)+1)]

	terminals_centroids = pd.concat([df_qgis_terminals_nodes.sh_terminals.apply(lambda x: x.coords[0]),
									 df_qgis_terminals.sh_terminals.apply(lambda x: x.centroid.coords[0])])

	df_qgis_pp_agg = df_pp_agg[['name', 'source', 'capacity(MW)', 'assigned_term']]
	df_qgis_pp_agg['lat'] = [terminals_centroids[x][0] for x in df_qgis_pp_agg.assigned_term]
	df_qgis_pp_agg['lon'] = [terminals_centroids[x][1] for x in df_qgis_pp_agg.assigned_term]
	df_qgis_pp_agg['sh_pp_agg'] = [Point(lat, lon) for lat,lon in zip(df_qgis_pp_agg.lat, df_qgis_pp_agg.lon)]
	df_qgis_pp_agg['idx'] = ['SM_{0:04}'.format(i) for i in range(1,len(df_pp_agg)+1)]

	df_qgis_lines.to_csv(write_data_dir + 'qgis_generated_csv/qgis_lines.csv', index=False)
	df_qgis_terminals.to_csv(write_data_dir + 'qgis_generated_csv/qgis_terminals.csv', index=False)
	df_qgis_terminals_nodes.to_csv(write_data_dir + 'qgis_generated_csv/qgis_terminals_nodes.csv', index=False)
	df_qgis_subs_og.to_csv(write_data_dir + 'qgis_generated_csv/qgis_subs_shape.csv', index=False)
	df_qgis_pp_og.to_csv(write_data_dir + 'qgis_generated_csv/qgis_power_plants.csv', index=False)
	df_qgis_pp_agg.to_csv(write_data_dir + 'qgis_generated_csv/qgis_power_plants_aggregated.csv', index=False)

	return 0


def main():

	print('--- EXECUTION STARTED ---')

	# print('Querying OSM for lines...')
	# df1 = query_overpass(lines.query_lines(query_area_tag, query_date))
	# print(df1)
	# df1.to_json(read_data_dir + '/DE_subregions/lines_DE_18072016.json', orient='records')
	# print('Querying OSM for terminals...')
	# df2 = query_overpass(terminals.query_terminals(query_area_tag, query_date))
	# df2.to_json(read_data_dir + '/DE_subregions/substations_DE_18072016.json', orient='records')
	# print('Querying OSM for power plants...')
	# df3 = query_overpass(power_plants.query_power_plants(query_area_tag, query_date))
	# df3.to_json(read_data_dir + '/DE_subregions/power_plants_DE_18072016.json', orient='records')

	print('Reading lines data from json file...')
	df1 = pd.read_json(read_data_dir + 'lines_DE_18072016.json', orient='records')
	print('Reading terminals data from json file...')
	df2 = pd.read_json(read_data_dir + 'substations_DE_18072016.json', orient='records')
	print('Reading power plants data from json file...')
	df3 = pd.read_json(read_data_dir + 'power_plants_DE_18072016.json', orient='records')

	print('Formatting data...')
	df_lines = lines.format_lines(df1)
	df_subs = terminals.format_substations(df2)
	df_pp = power_plants.format_power_plants(df3)

	print('Converting substations to circular areas...')
	df_subs = terminals.convert_substation_areas(df_subs, df_pp)

	print('Cleaning data...')
	df_lines = delete_lines_in_subs(df_lines, df_subs)
	print(' format data')
	df_lines = lines.handle_nan(df_lines)
	df_lines = lines.correct_incoherent_cables(df_lines)
	df_lines = lines.correct_incoherent_combinations(df_lines)
	print(' convert power plants capacity')
	df_pp = power_plants.convert_capacity(df_pp)
	print(' remove_redundant_generators')
	df_pp = power_plants.remove_redundant_generators(power_plants.cluster_data(df_pp, 1000), df_pp)

	print('Splitting lines by voltage...')
	df_lines = lines.split_lines_by_voltage(df_lines)

	print('Splitting lines for middle connections...')
	df_lines = lines.split_middle_connections(df_lines)

	print('Assigning terminals to lines...')
	df_lines, df_subs = assign_substation_to_lines(df_lines, df_subs)
	df_lines, df_subs = create_virtual_terminals(df_lines, df_subs)

	print('Handling lines with floating ends...')
	df_lines, df_subs = handle_floating_ends(df_lines, df_subs)

	print('Joining line segments...')
	df_lines = lines.join_lines(df_lines)

	print('Creating transformers data...')
	df_trafos = terminals.create_trafos_csv(df_subs)

	print('Handling cases still containing troublesome data...')
	df_lines, df_subs = handle_special_cases(df_lines, df_subs, df_trafos)

	print('Assigning terminals to power plants...')
	df_pp, df_pp_agg = assign_substation_to_power_plants(df_pp, df_subs)

	print('Writing data to csv...')
	create_pf_csv(df_lines, df_subs, df_pp_agg, df_trafos)

	print('Running tests...')
	tests.main_tests(df_lines.loc[df_lines.connecting_segments != 'chained_line'], df_subs, df_pp_agg)

	# print('Validating csv format...')
	# lines_csv_df = pd.read_csv(write_data_dir + 'csv/ElmLne.csv')
	# terminals_csv_df = pd.read_csv(write_data_dir + 'csv/ElmTerm.csv')
	# d = {'ElmLne': lines_csv_df, 'ElmTerm': terminals_csv_df}
	# validation = validate.NetworkValidator(d)
	# validation.validate()

	print('Creating files for QGIS visualization...')
	qgis(df_lines, df_subs, df_trafos, df2, df_pp, df_pp_agg)

	print('--- EXECUTION ENDED ---')

if __name__ == "__main__":
	main()