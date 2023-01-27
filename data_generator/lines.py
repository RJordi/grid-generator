import pandas as pd
import numpy as np
from haversine import haversine, Unit


def query_lines(query_area_tag, date) -> str:
    """
    Overpass API query to be run

    :param:
    :return: query string
    :rtype: str

    The query looks for lines and cables with a voltage value >= 220kV inside the area of Austria and filters out
    the lines and cables with a frequency of 16.7Hz, which belong to the railway system.
    """
    query = f"""
    // define output format
    [out:json]{date}[timeout:1000];

    // gather results
    area[{query_area_tag}]->.searchArea;
    (
      way["power"="line"](if:(t["voltage"]>=220000) && (t["frequency"]!=16.7))(area.searchArea);
      way["power"="cable"](if:(t["voltage"]>=220000) && (t["frequency"]!=16.7))(area.searchArea);
    );

    // print results
    out body geom;
    >;
      out tags;
    """
    return query


def calculate_line_length(coords_list: list) -> float:
    """
    Calculate the length of a line

    :param coords_list: Coordinates that define the line
    :type coords_list: list(list(float, float))
    :return: The line's length value
    :rtype: float
    """
    # Accumulated sum of the haversine distance between consecutive coordinates of a line element
    length = 0
    for i in range(1, len(coords_list)):
        length += haversine(coords_list[i-1], coords_list[i], unit=Unit.KILOMETERS)

    return length


def find_connecting_segments(row: pd.Series, df: pd.DataFrame) -> list:
    """
    Return the ids of the ways containing the initial or final node from the given list of nodes

    :param  pd.Series row: row of the Series consisting of the nodes that belong to a line
    :param  pd.DataFrame df: lines DataFrame
    :return: list with the ids of ways that contain the start or end node of row
    :rtype: list
    """
    # Get start and end node of each line element
    start_node = row.nodes[0]
    end_node = row.nodes[-1]

    # Find ids of line elements containing the start node (includes the one from the line elements we are evaluating)
    connecting_segments_start = df.loc[df.nodes.apply(lambda x: start_node in x)].index.tolist()
    # Delete own id
    connecting_segments_start.remove(row.name)
    # Filter out line elements that have different voltage from the evaluated line element
    connecting_segments_start = [seg for seg in connecting_segments_start if
                                 (row.name.split('_')[0] not in seg or 'seg' in row.name) and
                                 (df.voltage.loc[seg] == df.voltage.loc[row.name])]
    connecting_segments_end = df.loc[df.nodes.apply(lambda x: end_node in x)].index.tolist()
    connecting_segments_end.remove(row.name)  # delete own id
    connecting_segments_end = [seg for seg in connecting_segments_end if
                               (row.name.split('_')[0] not in seg or 'seg' in row.name) and
                               (df.voltage.loc[seg] == df.voltage.loc[row.name])]
    # Create a lists with the two lists
    connecting_segments = [connecting_segments_start] + [connecting_segments_end]

    return connecting_segments


def format_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format lines DataFrame

    :param pd.DataFrame df: lines DataFrame
    :return: formatted lines DataFrame
    :rtype: pd.DataFrame

    Get relevant information from tags and put it in the DataFrame's columns. Calculate the length of each line
    using :func:`calculate_line_length` and compute each line's connecting line using :func:`find_connecting_segments`.
    """
    # Drop duplicated entries
    df.drop_duplicates('id', inplace=True, ignore_index=True)

    # Set id of each element as the index of the DataFrame
    df.set_index(df.type.str[0]+df.id.astype(str), inplace=True, verify_integrity=True)
    df.index.name = 'id'

    df_lines = df.loc[df.type == 'way'][['nodes', 'geometry', 'tags']]

    # Delete lines with a frequency different from 50 Hz
    df_lines.drop(df_lines.loc[df_lines.tags.apply(lambda x: '50' not in x['frequency'] if 'frequency' in x.keys() \
        else False)].index, inplace=True)

    # Extract information from the "nodes" and "geometry" columns
    df_lines.geometry = df_lines.geometry.apply(lambda x: [[d['lat'], d['lon']] for d in x])
    df_lines.nodes = df_lines.nodes.apply(lambda x: ['n' + str(node_id) for node_id in x])

    # Create columns from the information in the "tags" column
    df_lines['name'] = df_lines.tags.apply(lambda x: x['name'] if 'name' in x.keys() else np.NaN)
    df_lines['voltage'] = df_lines.tags.apply(lambda x: x['voltage'] if 'voltage' in x.keys() else np.NaN)
    df_lines['line'] = df_lines.tags.apply(lambda x: x['line'] if 'line' in x.keys() else np.NaN)
    df_lines['cables'] = df_lines.tags.apply(lambda x: x['cables'] if 'cables' in x.keys() else np.NaN).astype(str)
    df_lines['cables'] = [sum([float(elem) for elem in cables_list]) for cables_list in df_lines.cables.str.split(';')]
    df_lines['circuits'] = df_lines.tags.apply(lambda x: x['circuits'] if 'circuits' in x.keys() else np.NaN).astype(str)
    df_lines['circuits'] = [sum([float(elem) for elem in circuits_list]) for circuits_list in df_lines.circuits.str.split(';')]
    df_lines['ref'] = df_lines.tags.apply(lambda x: x['ref'] if 'ref' in x.keys() else np.NaN)

    # Calculate length and connecting segments of each line element
    df_lines['length'] = df_lines.geometry.apply(calculate_line_length)

    # Drop "tags" column because it's no more useful and drop entries which don't have any digit in the "voltage" column
    df_lines.drop(columns=['tags'], inplace=True)
    df_lines.drop(df_lines.loc[df_lines.voltage.str.isalpha()].index, inplace=True)

    # Remove duplicated lines (different instances with same geometry)
    df_lines.drop(df_lines.loc[df_lines.geometry.apply(lambda x: tuple(x)).duplicated(keep='first')].index, inplace=True)

    return df_lines


def handle_nan(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN data in the DataFrame columns

    :param pd.DataFrame df_lines: lines pd.DataFrame
    :return: lines DataFrame without NaN values
    :rtype: pd.DataFrame
    """
    pd.options.mode.chained_assignment = None  # default='warn'

    # Handle nan in "cables" -> use number of references to determine number of cables
    nan_cables = df_lines.loc[df_lines.cables.isna()]
    nan_cables_known_ref = nan_cables.loc[nan_cables.ref.isna() == False]
    if len(nan_cables_known_ref) >= 1:
        df_lines.loc[nan_cables_known_ref.index, 'cables'] = nan_cables_known_ref.ref.str.split(';').str.len()*3

    # Use potentially known number of cables in the following line segment
    df_lines['connecting_segments'] = df_lines.nodes.to_frame().apply(lambda x: find_connecting_segments(x, df_lines), axis=1)
    nan_cables = df_lines.loc[df_lines.cables.isna()]  # Define "nan_cables" again as some entries have been solved

    def look_for_cables(segments: list, df: pd.DataFrame) -> float:
        if df.loc[segments[0]].cables.notna().any():
            return df.loc[segments[0]].cables.max()
        if df.loc[segments[1]].cables.notna().any():
            return df.loc[segments[1]].cables.max()

    df_lines.loc[nan_cables.index, 'cables'] = nan_cables.connecting_segments.apply(lambda x: look_for_cables(x, df_lines))

    # Set the cables to 3 for the rest of the entries (these entries have small length)
    nan_cables = df_lines.loc[df_lines.cables.isna()]
    df_lines.loc[nan_cables.index, 'cables'] = 3

    # Handle nan in "circuits"
    # -> use values in "cables" (if it is a multiple of 3) to calculate missing values in "circuits"
    df_lines.loc[(pd.isna(df_lines.circuits)) & (df_lines.cables%3 == 0), 'circuits'] = df_lines.cables/3

    # Handle nan in "name" -> fill nan values with an empty string
    df_lines.name.fillna(' ', inplace=True)

    return df_lines


def correct_incoherent_cables(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Correct incoherent data in the cables column

    :param pd.DataFrame df_lines: lines DataFrame
    :return: lines DataFrame without incoherent cables
    :rtype: pd.DataFrame

    The number of cables should always be a multiple of 3 because we are considering that all circuits are 3-phased
    """
    # Case where the references are known
    incoherent_cables = df_lines.loc[(df_lines.cables % 3 != 0) & (df_lines.ref.isna() == False)]
    if len(incoherent_cables) >= 1:
        df_lines.loc[incoherent_cables.ref.index, 'cables'] = incoherent_cables.ref.str.split(';').str.len()*3

    # Case where references are unknown -> the rest of lines have very short lengths, we give them a "cables" value of 3
    incoherent_cables = df_lines.loc[df_lines.cables % 3 != 0]
    if len(incoherent_cables) >= 1:
        df_lines.loc[incoherent_cables.index, 'cables'] = 3

    # Use values in "cables" (if it is a multiple of 3) to calculate missing values in "circuits"
    df_lines.loc[(pd.isna(df_lines.circuits)) & (df_lines.cables%3 == 0), 'circuits'] = df_lines.cables/3

    return df_lines


def correct_incoherent_combinations(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Correct incoherent data between columns

    :param pd.DataFrame df_lines: lines DataFrame
    :return: lines DataFrame without incoherent data
    :rtype: pd.DataFrame
    """
    # Number_of_circuits != Number_of_references
    if not df_lines.ref.isna().all():
        diff_ref_circuits = df_lines.loc[(df_lines.circuits != df_lines.ref.str.split(';').str.len()) & (df_lines.ref.isna() == False)]

        unlikely_circuits = diff_ref_circuits.loc[diff_ref_circuits.circuits >= 3]  # Lines with an unlikely number of circuits (>3 circuits)
        short_ways = diff_ref_circuits.loc[diff_ref_circuits.length < 0.2]  # Short lines (<0.2 km)
        long_ways = diff_ref_circuits.loc[diff_ref_circuits.length >= 0.2]  # Longer lines (>0.2 km)

        # Assign the number of references as the number of circuits for the "unlikely_circuits"
        df_lines.loc[unlikely_circuits.index, 'circuits'] = unlikely_circuits.ref.str.split(';').str.len()
        # Assign the minimum between the number of circuits and the number of references for short lines
        df_lines.loc[short_ways.index, 'circuits'] = pd.concat([short_ways.circuits, short_ways.ref.str.split(';').str.len()], axis=1).min(axis=1)
        # Assign the maximum between the number of circuits and the number of references for longer lines
        df_lines.loc[long_ways.index, 'circuits'] = pd.concat([long_ways.circuits, long_ways.ref.str.split(';').str.len()], axis=1).max(axis=1)

    # Number_of_cables != Number_of_circuits*3
    diff_cables_circuits = df_lines.loc[(df_lines.cables/3 != df_lines.circuits)]
    df_lines.loc[diff_cables_circuits.index, 'circuits'] = diff_cables_circuits.cables/3

    return df_lines


def split_lines_by_voltage(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Split lines with more than one voltage level into multiple lines (one for each voltage level)

    :param pd.DataFrame df_lines: lines DataFrame
    :return: lines DataFrame containing lines with a single voltage level per line
    :rtype: pd.DataFrame

    When the number of voltage levels is the same as the number of circuits a line is assigned to each voltage level
    (a line here is equivalent to a row of the DataFrame). When the number of voltage levels is different from the
    number of circuits we distinguish two cases:

    * **number_of_circuits % number_of_voltage_levels == 0**: we assign a line to each voltage level, but now the
      line has a number of circuits equal to: ::

        number_of_circuits / number_of_voltage_levels
    * **number_of_circuits % number_of_voltage_levels != 0**: we look if the references are known (there's one
      reference for each circuit):

      * **references are known**: we assign a line to each voltage level and the number of determined by the references
      * **references are not known**:  TODO
    """
    multi_volt_lines = df_lines.loc[df_lines.voltage.str.split(';').str.len() > 1]
    df_lines.drop(multi_volt_lines.index, inplace=True)

    # case where there are the same number of voltage levels as circuits
    same_circuits = multi_volt_lines.loc[multi_volt_lines.voltage.str.split(';').str.len() == multi_volt_lines.circuits]
    repeated_idxs = same_circuits.index.repeat(same_circuits.circuits).tolist()
    s_c_duplicated = same_circuits.reindex(repeated_idxs)

    def aux_func(row: pd.Series) -> pd.Series:
        num_duplicates_row = repeated_idxs.count(row.name)
        row.voltage = [volt for volt in row.voltage.split(";")][int(row.circuits - num_duplicates_row)]
        if len(str(row.ref).split(';')) == row.circuits:
            row.ref = [r for r in row.ref.split(";")][int(row.circuits - num_duplicates_row)]
        row.cables, row.circuits = 3, 1
        repeated_idxs.remove(row.name)
        return row

    s_c_duplicated = s_c_duplicated.apply(aux_func, axis=1)
    # Identify parallel lines with different voltages appending '_<voltage level>kV' to their index
    new_index = s_c_duplicated.index.astype(str) + '_' + (s_c_duplicated.voltage.astype(int)/1000).astype(int).astype(str) + 'kV'
    s_c_duplicated.set_index(new_index, inplace=True)
    # Drop line elements that have a voltage < 220kV
    s_c_duplicated.drop(s_c_duplicated.loc[s_c_duplicated.voltage.astype(int) < 220000].index, inplace=True)

    df_lines = pd.concat((df_lines, s_c_duplicated))

    # case where the number of voltage levels is different from the number of circuits

    mvl_diff_circ = multi_volt_lines.loc[multi_volt_lines.voltage.str.split(';').str.len() != multi_volt_lines.circuits]

    repeated_idxs = mvl_diff_circ.index.repeat(mvl_diff_circ.voltage.str.split(';').str.len()).tolist()
    diff_circ_duplicated = mvl_diff_circ.reindex(repeated_idxs)

        # case where number_of_circuits % number_of_voltage_levels == 0
    mvl_div = mvl_diff_circ.loc[mvl_diff_circ.circuits%mvl_diff_circ.voltage.str.split(';').str.len() == 0]

    volt = mvl_div.voltage.str.split(';').tolist()
    circuits = mvl_div.circuits.tolist()

    v_new = [elem for i in range(len(volt)) for elem in volt[i]]
    circ_new = [circuits[i]/len(volt[i]) for i in range(len(volt)) for e in volt[i]]
    cable_new = [circ*3 for circ in circ_new]
    diff_circ_duplicated.loc[mvl_div.index, 'voltage'] = v_new
    diff_circ_duplicated.loc[mvl_div.index, 'circuits'] = circ_new
    diff_circ_duplicated.loc[mvl_div.index, 'cables'] = cable_new

        # case where number_of_circuits % number_of_voltage_levels != 0
    mvl_non_div = mvl_diff_circ.loc[mvl_diff_circ.circuits%mvl_diff_circ.voltage.str.split(';').str.len() != 0]
    if len(mvl_non_div) >= 1:
                # case where the references are known and there are more references than voltage levels
        mvl_ref = mvl_non_div.loc[(mvl_non_div.ref.str.split(';').str.len() == mvl_non_div.circuits) & (mvl_non_div.ref.str.split(';').str.len() > mvl_non_div.voltage.str.split(';').str.len())]
        ref_first_num = [[ref[0] for ref in refs] for refs in mvl_ref.ref.str.split(';')]
        ref_first_num_unique = [list(dict.fromkeys(x)) for x in ref_first_num]
        ref_first_num_count = [ref_first_num[i].count(ref_first_num_unique[i][e]) for i in range(len(ref_first_num)) for e in range(len(ref_first_num_unique[i]))]
        volt = mvl_ref.voltage.str.split(';').explode().tolist()
        v_new = volt
        #v_new = [volt[i] for i in range(len(volt)) for x in range(ref_first_num_count[i])]
        circ_new = ref_first_num_count
        cable_new = [circ*3 for circ in circ_new]
        diff_circ_duplicated.loc[mvl_ref.index, 'voltage'] = v_new
        diff_circ_duplicated.loc[mvl_ref.index, 'circuits'] = circ_new
        diff_circ_duplicated.loc[mvl_ref.index, 'cables'] = cable_new
                # case where the references are known and there are less references than voltage levels
        mvl_ref_2 = mvl_non_div.loc[(mvl_non_div.ref.str.split(';').str.len() == mvl_non_div.circuits) & (mvl_non_div.ref.str.split(';').str.len() <= mvl_non_div.voltage.str.split(';').str.len())]
        v_new = mvl_ref_2.voltage.str.split(';').explode().tolist()
        diff_circ_duplicated.loc[mvl_ref_2.index, 'voltage'] = v_new
        diff_circ_duplicated.loc[mvl_ref_2.index, 'circuits'] = 1
        diff_circ_duplicated.loc[mvl_ref_2.index, 'cables'] = 3
                # case where the references are NOT known
        mvl_non_ref = mvl_non_div.loc[mvl_non_div.ref.str.split(';').str.len() != mvl_non_div.circuits]
        volt = mvl_non_ref.voltage.str.split(';').tolist()
        circuits = mvl_non_ref.circuits.tolist()
        floor_div = (mvl_non_ref.circuits//mvl_non_ref.voltage.str.split(';').str.len()).tolist()
        div_residual = (mvl_non_ref.circuits%mvl_non_ref.voltage.str.split(';').str.len()).tolist()
        v_new = [elem for i in range(len(volt)) for elem in volt[i]]
        circ_new = []
        for i in range(len(volt)):
            for elem in volt[i]:
                circ_new.append(floor_div[i] + div_residual[i])
        cable_new = [circ*3 for circ in circ_new]
        diff_circ_duplicated.loc[mvl_non_ref.index, 'voltage'] = v_new
        diff_circ_duplicated.loc[mvl_non_ref.index, 'circuits'] = circ_new
        diff_circ_duplicated.loc[mvl_non_ref.index, 'cables'] = cable_new

    # Identify parallel lines with different voltages appending '_<voltage level>kV' to their index
    new_index = diff_circ_duplicated.index.astype(str) + '_' + (diff_circ_duplicated.voltage.astype(int)/1000).astype(int).astype(str) + 'kV'
    diff_circ_duplicated.set_index(new_index, inplace=True)
    # Drop line elements that have a voltage < 220kV
    diff_circ_duplicated.drop(diff_circ_duplicated.loc[diff_circ_duplicated.voltage.astype(int) < 220000].index, inplace=True)

    df_lines = pd.concat((df_lines, diff_circ_duplicated))

    # Join duplicated elements (same voltage id and level) by adding their circuits
    idx_counts = df_lines.index.value_counts()
    df_lines = df_lines.groupby(df_lines.index).first()
    df_lines.loc[idx_counts > 1, 'circuits'] = idx_counts.loc[idx_counts > 1].tolist()
    df_lines.loc[idx_counts > 1, 'cables'] = df_lines.loc[idx_counts > 1].circuits*3

    # Format voltage value
    df_lines.voltage = df_lines.voltage.astype('float')/1000
    round_voltage = df_lines.loc[~df_lines.voltage.isin([220.0, 380.0])].voltage
    new_voltage = [min([220.0, 380.0], key=lambda x:abs(x-volt)) for volt in round_voltage]
    df_lines.loc[~df_lines.voltage.isin([220.0, 380.0]), 'voltage'] = new_voltage
    # Calculate connecting segments again as we have changed the ids of some entries
    df_lines['connecting_segments'] = df_lines.nodes.to_frame().apply(lambda x: find_connecting_segments(x, df_lines), axis=1)
    # Put index into "id" column
    df_lines['id'] = df_lines.index

    return df_lines


def split_middle_connections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split lines that are connected to other lines at some point which is not its start or end

    :param ps.DataFrame df: lines DataFrame
    :return: lines DataFrame without any middle connection
    :rtype: pd.DataFrame

    TODO -> More detailed explanation of what the function does
    """
    l = [any([len(seg) == 1 for seg in x]) for x in df.connecting_segments]
    df_l = df.loc[l]
    lines_to_split, where_to_split = [], []

    connecting_segments_col = df_l.connecting_segments.to_numpy()
    nodes_col = df_l.nodes.to_numpy()
    nodes_col_all = df.nodes.to_numpy()
    idx_col_all = df.index.to_numpy()
    for e in range(len(connecting_segments_col)):
        i = 0
        for segment in connecting_segments_col[e]:
            if len(segment) == 1:
                connecting_line_id = segment[0]
                connecting_line_id_idx = np.where(idx_col_all==connecting_line_id)[0][0]
                connecting_node = nodes_col[e][-i]
                if connecting_node not in [(nodes_col_all[connecting_line_id_idx])[0], (nodes_col_all[connecting_line_id_idx])[-1]]:
                    lines_to_split.append(connecting_line_id)
                    where_to_split.append(connecting_node)
            i += 1

    df_lines_to_split = df.loc[lines_to_split]

    index_where_to_split = [df_lines_to_split.nodes[i].index(where_to_split[i]) for i in range(len(lines_to_split))]

    # Handle cases where there is more than ONE line connecting to a middle node
    df_lines_to_split['where_to_split'] = where_to_split
    df_lines_to_split['idx_where_to_split'] = index_where_to_split
    df_lines_to_split = df_lines_to_split.groupby('id').agg({'where_to_split': lambda x: x.tolist(),
                                                             'idx_where_to_split': lambda x: sorted(x.tolist())})

    nodes_col_split, geometry_col_split = df.loc[df_lines_to_split.index].nodes.tolist(), df.loc[df_lines_to_split.index].geometry.tolist()
    splitted_nodes, splitted_geometry = [], []
    idxs = df_lines_to_split.idx_where_to_split.tolist()

    for i in range(len(idxs)):
        splitted_nodes += [nodes_col_split[i][:idxs[i][0]+1]] + [nodes_col_split[i][idxs[i][x]:idxs[i][x+1]+1] for x in range(len(idxs[i])-1)] + [nodes_col_split[i][idxs[i][-1]:]]
        splitted_geometry += [geometry_col_split[i][:idxs[i][0]+1]] + [geometry_col_split[i][idxs[i][x]:idxs[i][x+1]+1] for x in range(len(idxs[i])-1)] + [geometry_col_split[i][idxs[i][-1]:]]

    new_index = [df_lines_to_split.index[i] + '_seg_' + str(e) for i in range(len(df_lines_to_split)) for e in range(1,len(idxs[i])+2)]

    df_to_concat = pd.DataFrame(list(zip(splitted_nodes, splitted_geometry)), index=new_index, columns =['nodes', 'geometry'])
    new_index_id = ['_'.join(idx.split('_')[:-2]) for idx in new_index]
    df_to_concat['name'] = df.name.loc[new_index_id].tolist()
    df_to_concat['voltage'] = df.voltage.loc[new_index_id].tolist()
    df_to_concat['line'] = df.line.loc[new_index_id].tolist()
    df_to_concat['cables'] = df.cables.loc[new_index_id].tolist()
    df_to_concat['circuits'] = df.circuits.loc[new_index_id].tolist()
    df_to_concat['ref'] = df.ref.loc[new_index_id].tolist()
    df_to_concat['length'] = df_to_concat.geometry.apply(calculate_line_length)

    df.drop(df_lines_to_split.index, inplace=True)
    df_lines = pd.concat([df, df_to_concat])
    df_lines['connecting_segments'] = df_lines.nodes.to_frame().apply(lambda x: find_connecting_segments(x, df_lines), axis=1)
    df_lines['id'] = df_lines.index

    # Set new index
    new_index = pd.Index([f'ElmLne_{i}' for i in range(1,len(df_lines)+1)])
    df_lines.set_index(new_index, inplace=True)

    return df_lines


def join_lines(df:pd.DataFrame) -> pd.DataFrame:
    """
    Join two line elements that are exclusively connected between each other

    :param pd.DataFrame df: lines DataFrame
    :return: lines DataFrame with concatenated lines
    :rtype: pd.DataFrame

    TODO: Detailed explanation
    """
    from ast import literal_eval

    l = [any([pd.isna(x[i]) for i in [0,1]]) for x in df.substations]
    df_join = df.loc[l]
    line_ids = df_join.id.tolist()
    line_connections = df_join.connecting_segments.tolist()
    lines_subs = df_join.substations.tolist()
    final_list = []

    def recursive_join(i:int, j_list:list) -> list:
        for segment in line_connections[i]:
            if len(segment) == 1 and segment[0] not in j_list and pd.isna(lines_subs[i][line_connections[i].index(segment)]):
                j_list.append(segment[0])
                try:
                    recursive_join(line_ids.index(segment[0]), j_list)
                except:
                    pass
        return j_list

    for i in range(len(line_ids)):
        join_list = [line_ids[i]]
        final_list.append(sorted(recursive_join(i, join_list))) #why do I sort the list?

    ser_chains = pd.Series(final_list).drop_duplicates().reset_index(drop=True)
    ser_voltages = pd.Series([[df.voltage.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_terminals = pd.Series([[df.substations.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])

    inconsistent_voltages = []
    inconsistent_terminals = []
    inconsistent_terminals_2 = []
    for i in range(len(ser_chains)):
        if len(set(ser_voltages[i])) > 1:
            inconsistent_voltages.append(i)
            #print(i, ser_chains[i], ser_terminals[i])
        if any([np.nan not in seg for seg in ser_terminals[i]]):
            inconsistent_terminals.append(i)
        if [pd.isna(x) == False for seg in ser_terminals[i] for x in seg].count(True) != 2:
            inconsistent_terminals_2.append(i)

    # print('inconsistent_voltages: ', inconsistent_voltages, 'len = ', len(inconsistent_voltages))
    # print('inconsistent_terminals: ', inconsistent_terminals, 'len = ', len(inconsistent_terminals))
    # print('Examples: ', ser_chains[inconsistent_terminals[7:12]], ser_terminals[inconsistent_terminals[7:12]])
    # print('inconsistent_terminals_2: ', inconsistent_terminals_2, 'len = ', len(inconsistent_terminals_2))

    # Create new entry in df for each list in series and drop the entries that appear in the list
    ser_voltages = pd.Series([[df.voltage.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_circuits = pd.Series([[df.circuits.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_terminals = pd.Series([[df.substations.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_names = pd.Series([[df.name.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_lengths = pd.Series([[df.length.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_nodes = pd.Series([[df.nodes.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_geometry = pd.Series([[df.geometry.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])
    ser_ids = pd.Series([[df.id.loc[df.id == elem].values[0] for elem in chain] for chain in ser_chains])

    # Order coordinates to chain segments in order
    ordered_ser_geometry = []
    for ser_geom_elem in ser_geometry:
        start_end_points = [c_list[::len(c_list)-1] for c_list in ser_geom_elem]
        start_end_points = [point for elem in start_end_points for point in elem]
        unique_start_end_points = pd.Series(start_end_points).drop_duplicates(keep=False).tolist()
        #assert len(unique_start_end_points) == 2, (ser_geom_elem, start_end_points,unique_start_end_points)
        if len(unique_start_end_points) != 2:
            # print(ser_chains[ser_geometry.tolist().index(ser_geom_elem)])
            ordered_ser_geometry.append(ser_geom_elem)
            continue
        start_point = unique_start_end_points[0]
        end_point = unique_start_end_points[1]
        #print('Start point: ', start_point, '\n')

        def recursive_func(s_p, new_list):
            for c_list in ser_geom_elem:
                if s_p in c_list:
                    if c_list.index(s_p) == len(c_list)-1:
                        c_list.reverse()
                    #print(ser_geom_elem[ser_geom_elem.index(c_list)], '\n')
                    del ser_geom_elem[ser_geom_elem.index(c_list)]
                    new_start = c_list[-1]
                    new_list.append(c_list)
                    break
#             print('Updated start point \n', new_start, '\n')
#             print('Updated ser_geom_elem \n', ser_geom_elem, '\n')
#             print('Updated new_list \n', new_list, '\n')
            if ser_geom_elem != []:
                recursive_func(new_start, new_list)
            return new_list

        new_geom_elem = recursive_func(start_point, [])
        #print('new_geom_elem: ', new_geom_elem)
        ordered_ser_geometry.append(str(new_geom_elem))

    ser_geometry = pd.Series(ordered_ser_geometry).astype(str)
    ser_geometry = ser_geometry.apply(lambda x: literal_eval(x))

    # Create dataframe for the chained lines
    df_chains = pd.DataFrame(columns=['nodes','geometry','name','voltage','line','cables','circuits','ref','length','connecting_segments','id','substations'])
    df_chains['name'] = pd.Series([next((name for name in elem if name != ' '), ' ') for elem in ser_names])
    df_chains['nodes'] = pd.Series([[node for n_list in ser_nodes_elem for node in n_list] for ser_nodes_elem in ser_nodes])
    df_chains['geometry'] = pd.Series([[coords for c_list in ser_geom_elem for coords in c_list] \
                                       for ser_geom_elem in ser_geometry])
    df_chains['voltage'] = pd.Series([max(voltages) for voltages in ser_voltages])
    df_chains['circuits'] = pd.Series([max(circuits) for circuits in ser_circuits])
    df_chains['length'] = pd.Series([sum(lengths) for lengths in ser_lengths])
    df_chains['substations'] = [[x for term in elem for x in term if pd.isna(x)==False] for elem in ser_terminals]
    df_chains['id'] = pd.Series([','.join(ids) for ids in ser_ids])
    df_chains['connecting_segments'] = pd.Series(['chained_line' for chain in ser_chains])

    max_elm_lines_idx = max([int(idx.split('_')[1]) for idx in df.index])
    new_index_df_chains = pd.Index([f'ElmLne_{i + max_elm_lines_idx}' for i in range(1,len(df_chains)+1)])
    df_chains.set_index(new_index_df_chains, inplace=True)

    df.drop(df_join.index, inplace=True)
    df = pd.concat([df, df_chains])

    return df
