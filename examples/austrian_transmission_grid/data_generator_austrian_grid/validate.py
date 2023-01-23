"""
Validate grid data for load flow calculations
"""
import logging
import re

import pandera as pa
from pandas.api.types import is_number as pd_is_number

from tabulate import tabulate

LOGGER = logging.getLogger(__name__)

SUPPORTED_SCHEMAS = [
    "ElmGenstat",
    "ElmLne",
    "ElmLod",
    "ElmTerm",
    "ElmXnet",
    "TypLne",
    "StaPqmea"
]

# https://github.com/pandera-dev/pandera/issues/466
is_number = pa.Check(lambda s: s.map(pd_is_number), name="is_number")

# GPS regex pattern
gps_regex_pattern = r'(\[)(\[\d*\.\d*\,\s\d*\.\d*\])(\,\s\[\d*\.\d*\,\s\d*\.\d*\])*(\])'

# TODO: this pattern does find \n character, but in
# tests/test_validate.py::test_loc_name_invalid_characters
# it is not listed
LOC_NAME_INVALID_CHARACTERS = re.compile(r'[:*?=",~|\r\n\\]')


def invalid_characters(s) -> bool:
    # note that nan != nan
    return True if s != s else not re.findall(LOC_NAME_INVALID_CHARACTERS, s)


loc_name = pa.Column(
    str,
    checks=[
        pa.Check(lambda x: invalid_characters(x), element_wise=True),
    ],
    unique=True,
)

loc_name_not_unique = pa.Column(
    str,
    checks=[
        pa.Check(lambda x: invalid_characters(x), element_wise=True),
    ],
    unique=False,
)

ElmGenstat_schema = pa.DataFrameSchema(
    name="ElmGenstat",
    columns={
        "loc_name": loc_name,
        "bus1": pa.Column(int),
        "cCategory": pa.Column(str, required=False),
        "sgn": pa.Column(float, required=False),
        "pgini": pa.Column(float, required=False),
        "cosn": pa.Column(checks=is_number, required=False),
    },
    coerce=True,
)

ElmArea_schema = pa.DataFrameSchema(
    name="ElmArea",
    columns={
        "loc_name": loc_name,
    },
    coerce=True,
)

ElmLne_schema = pa.DataFrameSchema(
    name="ElmLne",
    columns={
        "loc_name": loc_name,
        "bus1.cterm": pa.Column(str, required=False),
        "bus2.cterm": pa.Column(str, required=False),
        "dline": pa.Column(float, checks=pa.Check.greater_than(0)),
        "typ_id": pa.Column(str, required=False, nullable=False),
        "inAir": pa.Column(int, required=False),
        "nlnum": pa.Column(int, required=False),
        "outserv": pa.Column(int, required=False),
        "GPScoords": pa.Column(str, pa.Check.str_matches(gps_regex_pattern), required=False),
        "desc": pa.Column(str, required=False),
    },
    coerce=True,
)

ElmLod_schema = pa.DataFrameSchema(
    name="ElmLod",
    columns={
        "loc_name": loc_name,
        "bus1": pa.Column(str),
        "plini": pa.Column(float),
        "coslini": pa.Column(checks=is_number, required=False),
        "classif": pa.Column(str, required=False),
        "desc": pa.Column(str, required=False),
    },
    coerce=True,
)

ElmTerm_schema = pa.DataFrameSchema(
    name="ElmTerm",
    columns={
        "loc_name": loc_name,
        "uknom": pa.Column(float, checks=pa.Check.greater_than(0)),
        "phtech": pa.Column(int, required=False),
        "systype": pa.Column(int, required=False),
        "desc": pa.Column(str, required=False),
    },
    coerce=True,
)

ElmXnet_schema = pa.DataFrameSchema(
    name="ElmXnet",
    columns={
        "loc_name": loc_name,
        "bus1": pa.Column(str),
        "bustp": pa.Column(str, required=False),
        "rntxn": pa.Column(float, required=False),
        "snss": pa.Column(float, nullable=True, required=False),
    },
    coerce=True,
)

ElmTemplate_schema = pa.DataFrameSchema(
    name="ElmTemplate",
    columns={
        "loc_name": loc_name,
        "template": pa.Column(str),
        "cterm": pa.Column(str)
    },
    coerce=True,
)

TypLne_schema = pa.DataFrameSchema(
    name="TypLne",
    columns={
        "loc_name": loc_name,
        "uline": pa.Column(float),
        "sline": pa.Column(float),
        "InomAir": pa.Column(float, required=False),
        "cohl_": pa.Column(checks=is_number),
        "rline": pa.Column(float),
        "xline": pa.Column(float),
        "rline0": pa.Column(checks=is_number, required=False),
        "xline0": pa.Column(checks=is_number, required=False),
        "systp": pa.Column(int, required=False),
        "nlnph": pa.Column(int, required=False),
        "nneutral": pa.Column(int, required=False),
        "frnom": pa.Column(checks=is_number, required=False),
        "desc": pa.Column(str, required=False),
    },
    coerce=True,
)

StaPqmea_schema = pa.DataFrameSchema(
    name="StaPqmea",
    columns={
        "loc_name": pa.Column(str, checks=[pa.Check(lambda x: invalid_characters(x), element_wise=True)], unique=False),
        "pcubic": pa.Column(str),
        "i_orient": pa.Column(int, required=False),
        "nphase": pa.Column(int, required=False),
        "ElmComp": pa.Column(str),
    },
    coerce=True,
)

StaVmea_schema = pa.DataFrameSchema(
    name="StaVmea",
    columns={
        "loc_name": pa.Column(str, checks=[pa.Check(lambda x: invalid_characters(x), element_wise=True)], unique=False),
        "pbusbar": pa.Column(str),
        "i_orient": pa.Column(int, required=False),
        "nphase": pa.Column(int, required=False),
        "ElmComp": pa.Column(str),
    },
    coerce=True,
)


class NetworkValidator:
    def __init__(self, d: dict):
        self.d = d
        self.schemas = {
            "ElmGenstat": ElmGenstat_schema,
            "ElmLne": ElmLne_schema,
            "ElmLod": ElmLod_schema,
            "ElmTerm": ElmTerm_schema,
            "ElmXnet": ElmXnet_schema,
            "ElmTemplate": ElmTemplate_schema,
            "TypLne": TypLne_schema,
            "StaPqmea": StaPqmea_schema,
            "StaVmea": StaVmea_schema
        }

    def validate(self):
        for k, v in self.d.items():
            if k in SUPPORTED_SCHEMAS:
                try:
                    self.schemas[k].validate(v, lazy=True)
                except pa.errors.SchemaErrors as err:
                    LOGGER.error('Error validation input data! Please check the report below.')
                    print("Schema errors and failure cases:")
                    print(tabulate(err.failure_cases, headers='keys', tablefmt='psql', showindex=False))
                    #print("\nDataFrame object that failed validation:")
                    #print(tabulate(err.data, headers='keys', tablefmt='psql', showindex=False))
                    raise err

    def validate_line_types(self):
        if not self.d.keys() & {"TypLne", "ElmLne"}:
            LOGGER.warning("did not find both TypLne and ElmLne")
        line_type_ids = list(self.d["TypLne"]["loc_name"].unique())
        line_type_schema = pa.DataFrameSchema(
            name="ElmLne+TypLne",
            columns={
                "typ_id": pa.Column(checks=pa.Check.isin(line_type_ids)),
            },
        )
        line_type_schema.validate(self.d["ElmLne"], lazy=True)
