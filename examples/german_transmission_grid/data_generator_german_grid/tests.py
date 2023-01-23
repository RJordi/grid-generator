import pandas as pd
import numpy as np

def test_duplicated_elements(df_lines:pd.DataFrame, df_terminals:pd.DataFrame):
    """
    Check that there aren't any duplicated lines or duplicated terminals in the corresponding DataFrames
    """
    duplicated_lines = pd.Series([geom+[volt] for geom,volt in zip(df_lines.geometry, df_lines.voltage)]).duplicated(keep=False).any()
    assert duplicated_lines == False, 'ASSERTION FAILED: There are duplicated lines'

    duplicated_terminals = pd.Series([geom+[volt] for geom,volt in zip(df_terminals.geometry, df_terminals.voltage)]).duplicated(keep=False).any()
    assert duplicated_terminals == False, 'ASSERTION FAILED: There are duplicated terminals'


def test_transmission_voltage_levels(df_lines:pd.DataFrame, df_terminals:pd.DataFrame):
    """
    Check that the unique voltage levels of the lines and the terminals are the same, and
    they correspond to the transmission power system
    """
    lines_voltage_levels = sorted([v for v in df_lines.voltage.unique()])
    terminals_voltage_levels = sorted([v for v in df_terminals.voltage.unique()])
    assert lines_voltage_levels == terminals_voltage_levels and all([v_level>100 for v_level in lines_voltage_levels]), 'ASSERTION FAILED: There are incorrect voltage levels'


def test_line_ends_different_terminals(df_lines:pd.DataFrame):
    """
    Check that a line doesn't start and end at the same terminal
    """
    check = any([term[0] == term[1] for term in df_lines.substations])
    assert check == False, 'ASSERTION FAILED: There are lines starting and ending at the same terminal'

def test_line_ends_nowhere(df_lines:pd.DataFrame):
    """
    Check that there aren't any lines with "floating" ends
    """
    floating_lines = any([any([x[i]==[] and pd.isna(y[i]) for i in range(0,2)]) for x,y in zip(df_lines.connecting_segments, df_lines.substations)])
    assert floating_lines == False, 'ASSERTION FAILED: There are lines with ends not connected to any terminal ("floating ends")'

def test_virtual_terminal_is_correct(df_lines:pd.DataFrame):
    """
    Check that when there is a virtual terminal, the number of connecting segments is always >= 2 
    """
    check = any([any([len(conn_seg[i])<2 if (pd.isna(terms[i])==False and 'virtual' in terms[i]) else False for i in [0,1]]) for terms,conn_seg in zip(df_lines.substations, df_lines.connecting_segments)])
    assert check == False, 'ASSERTION FAILED: There are virtual terminals in line ends with only one connecting segment'

def test_terminal_voltages(df_lines:pd.DataFrame, df_terminals:pd.DataFrame):
    """
    Check that the terminals to which a line is connected, have the same voltage level as the line
    """
    check = any([any([line_volt != df_terminals.voltage.loc[terms[i]] if (pd.isna(terms[i])==False) else False for i in [0,1]]) for terms,line_volt in zip(df_lines.substations, df_lines.voltage)])
    assert check == False, 'ASSERTION FAILED: There are lines with a voltage level not corresponding to the ones from their terminals'


def test_nan_values_in_terminals(df_lines:pd.DataFrame):
    """
    Check that all lines end at a terminal at both of its ends
    """
    check = df_lines.terminals.explode().isnull().any()
    assert check == False,  "ASSERTION FAILED: There are lines that don't have a terminal assigned to one of its ends"

def test_terminals_exist(df_lines:pd.DataFrame, df_terminals:pd.DataFrame, df_pp:pd.DataFrame):
    """
    Check that all terminals to which a line or a power plant is connected to exist, i.e., it appears in the terminals DataFrame
    """
    check = any([term not in df_terminals.index.tolist() for term in np.unique(np.array(df_lines.substations.explode().unique().tolist() + df_pp.assigned_term.tolist()))])
    assert check == False, "ASSERTION FAILED: Some elements have a nonexistent assigned terminal"


def main_tests(df_lines:pd.DataFrame, df_terminals:pd.DataFrame, df_pp:pd.DataFrame):
    """
    Run some tests to make sure data is coherent and doesn't contain fatal errors

    :param pd.DataFrame df_lines: lines DataFrame
    :param pd.DataFrame df_terminals: terminals DataFrame
    :param pd.DataFrame df_pp: power plants DataFrame
    """
    test_duplicated_elements(df_lines, df_terminals)
    test_line_ends_different_terminals(df_lines)
    test_line_ends_nowhere(df_lines)
    test_virtual_terminal_is_correct(df_lines)
    test_terminal_voltages(df_lines, df_terminals)
    test_transmission_voltage_levels(df_lines, df_terminals)
    test_terminals_exist(df_lines, df_terminals, df_pp)
    print('Tests passed successfully!')