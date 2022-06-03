"""
This module contains the exceptions for the CLI.
"""


class UnsupportedFileFormat(Exception):
    pass


class MissingArgument(Exception):
    pass


class LabelColumnNotFound(Exception):
    pass
