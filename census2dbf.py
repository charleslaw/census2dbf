#!/usr/bin/env python
# encoding: utf-8
"""
csv2dbf.py
Created by Neil Freeman on 2012-06-10.
Licensed under Creative Commons 3.0 BY SA (aka CC 3.0 BY SA).
http://creativecommons.org/licenses/by-sa/3.0/
"""

import sys
import argparse
import csv
import re
import struct
import datetime
import itertools
from operator import itemgetter

help_message = '''Convert US Census CSVs to DBF.'''


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def collect(l, index):
    return map(itemgetter(index), l)


def dbfwriter(f, fieldnames, fieldspecs, records):
    """
    source: http://code.activestate.com/recipes/362715-dbf-reader-and-writer/
    From activestate TOS:
    All Content deposited shall to be governed by the Creative Commons 3.0 BY SA (aka CC 3.0 BY SA).
    This license can be found at http://creativecommons.org/licenses/by-sa/3.0/
    """

    """ Return a string suitable for writing directly to a binary dbf file.

    File f should be open for writing in a binary mode.

    Fieldnames should be no longer than ten characters and not include \x00.
    Fieldspecs are in the form (type, size, deci) where
        type is one of:
            C for ascii character data
            M for ascii character memo data (real memo fields not supported)
            D for datetime objects
            N for ints or decimal objects
            L for logical values 'T', 'F', or '?'
        size is the field width
        deci is the number of decimal places in the provided decimal object
    Records can be an iterable over the records (sequences of field values).

    """
    # header info
    ver = 3
    now = datetime.datetime.now()
    yr, mon, day = now.year - 1900, now.month, now.day
    numrec = len(records)
    numfields = len(fieldspecs)
    lenheader = numfields * 32 + 33
    lenrecord = sum(field[1] for field in fieldspecs) + 1
    hdr = struct.pack('<BBBBLHH20x', ver, yr, mon, day, numrec, lenheader, lenrecord)
    f.write(hdr)

    # field specs
    for name, (typ, size, deci) in itertools.izip(fieldnames, fieldspecs):
        name = name.ljust(11, '\x00')
        fld = struct.pack('<11sc4xBB14x', name, typ, size, deci)
        f.write(fld)

    # terminator
    f.write('\r')

    # records
    for record in records:
        f.write(' ')                        # deletion flag
        for (typ, size, deci), value in itertools.izip(fieldspecs, record):
            if typ == "N":
                value = str(value).rjust(size, ' ')
            elif typ == 'D':
                value = value.strftime('%Y%m%d')
            elif typ == 'L':
                value = str(value)[0].upper()
            else:
                value = str(value)[:size].ljust(size, ' ')
            assert len(value) == size
            f.write(value)

    # End of file
    f.write('\x1A')


class csv_parser(object):
    # open up the csv, decide what kind of format it is, and get the fields and data ready.
    # input: file handle
    # input (optional): field types, date format
    # output: fieldnames, lengths, types

    def __init__(self, handle):
        self.handle = handle
        self.reader = csv.reader(self.handle)

    def parse_header(self):
        'parses a row and returns fieldnames'
        full_header, on_header, i = [], True, 0

        self.handle.seek(0)

        # Handle the first row as the header
        row = next(self.reader)
        fieldnames = [f[:11] for f in row]  # Clip each field to 10 characters.
        if fieldnames[0] == '':
            fieldnames[0] = 'geoid'
        if fieldnames[1] == '':
            fieldnames[1] = 'geoid2'
        if fieldnames[2] == '':
            fieldnames[2] = 'geoname'
        # Replace illegal characters in fieldnames with underscore.
        illegal = re.compile(r'[^\w]')
        fieldnames = [illegal.sub('', x).lower() for x in fieldnames]

        full_header.append(fieldnames)

        while on_header is True:
            row = next(self.reader)

            i += 1
            # pull the fieldnames from the first row of headers
            if re.match(r'\w{7}US\d+', row[0]):
                types = [self.get_field_type(f) for f in row]
                lengths = [len(x) for x in row]
                on_header = False
            else:
                full_header.append(row)

        full_header = zip(*full_header)
        deci = [''] * len(types)

        # Pointer doesn't move
        header, field_specs = self.spec_fields(fieldnames, types, lengths, deci)

        return field_specs, i, header, full_header

    def get_field_type(self, value):
        'Value should be a string read from the csv.'
        'Will return value in the (possibly) correct type.'

        try:
            if float(value) == int(value):
                return int
            else:
                return float
        except ValueError:
            pass

        # Return string as an error
        return str

    def spec_fields(self, names, types, lengths, decis):
        # fieldspecs are going to look like: [(C', x), ('N', y, 0)]
        field_specs = zip(names, types, lengths, decis)
        new_field_specs, header = [], []

        for name, typ, length, deci in field_specs:
            if (name == 'geoid' or name == 'geoid2' or typ == str):
                typ, deci = 'C', 0
            elif typ == int:
                typ, deci = 'N', 0
            elif typ == float:
                typ, deci = 'N', 2

            new_field_specs.append(tuple([typ, length, deci]))
            header.append(name)

        return header, new_field_specs

    def get_field_lengths(self, field_specs):
        types, lengths, decis = zip(*field_specs)
        lengths = list(lengths)
        for row in self.reader:
            e = enumerate(row)
            for k, field in e:
                lengths[k] = max(lengths[k], len(field))

        return zip(list(types), lengths, list(decis))

    def point_at_data(self):
        self.handle.seek(0)
        for i in range(self.num_headers):
            self.reader.next()

    def get_records(self):
        #place pointer in the right place
        self.point_at_data()
        return [row for row in self.reader]

    def parse(self):
        field_specs, self.num_headers, header, self.datadict = self.parse_header()
        # Pointer is on first data row.
        field_specs = self.get_field_lengths(field_specs)
        # Pointer is at end.
        # This resets pointer back to where data starts
        self.records = self.get_records()

        return header, field_specs, self.records

    def write_dd(self, input_file, output_file):
        now = datetime.datetime.now()
        output = new_file_ending(output_file.name, "-data-dictionary.txt")

        with open(output, 'w+') as data_dict_handle:
            msg = "Data Dictionary\nAutomatically extracted from the header of {0}\n{1}\n".format(input_file, now.strftime("%Y-%m-%d %H:%M"))
            data_dict_handle.write(msg)
            out_list = []
            for row in self.datadict:
                out_str = ""
                for field in row:
                    out_str += field + "\t"

                out_list.append(out_str + "\n")
            data_dict_handle.writelines(out_list)


def new_file_ending(filename, new):
    try:
        k = filename.rindex('.')
        return filename[0:k] + new
    except:
        return filename + new


def convert_csv_to_dbf(input_file, output_file, mapping_dbf,
                       mapping_from, mapping_to, print_data_dict=False):
    input_file = open(input_file, 'r')
    output_file = open(output_file, 'w')
    mapping_file = open(mapping_dbf, 'r')
    _convert_csv_to_dbf(input_file, output_file, mapping_file,
                        mapping_from, mapping_to, print_data_dict=print_data_dict)


def _find_field_index_dbf(fields, name):
    """
    Return the index of the field with the given name
    If not found, raise an exception
    """
    for i, field_info in enumerate(fields):
        if field_info[0] == name:
            return i
    raise ValueError('Field %s not found in data' % name)


def _convert_csv_to_dbf(input_file, output_file, mapping_file=None,
                        mapping_from=None, mapping_to=None, print_data_dict=False):
    if output_file is None:
        name = new_file_ending(input_file.name, '.dbf')
        output_file = open(name, 'w')

    if mapping_file:
        #Read this file and map it
        try:
            from shapefile import Reader
        except ImportError:
            print "pyshp required for mapping feature"
            raise
        dbfr = Reader(dbf=mapping_file)
        #find field thath as the mapping_from name
        #use -1 because pyshp adds a column for flagging deleted fields
        name_i = _find_field_index_dbf(dbfr.fields, mapping_from) - 1
        map_values = [rec[name_i] for rec in dbfr.iterRecords()]

    # Parse the csv.
    parser = csv_parser(handle=input_file)
    header, fieldspecs, records = parser.parse()
    if mapping_file:
        csv_name_i = header.index(mapping_to)
        #be conservative and make sure they match
        if len(records) != len(map_values):
            raise Exception('mapping records lengths must match')
        #reorder the records so they match the original
        #This will reaise an error if something does not map
        mapped_records = [None]*len(map_values)
        for i in xrange(len(map_values)):
            mv = map_values[i]
            try:
                old_i = collect(records, csv_name_i).index(mv)
            except ValueError:
                raise ValueError('Could not find record name %s in csv' % mv)
            mapped_records[i] = records[old_i]
        records = mapped_records

    # Write to dbf.
    dbfwriter(output_file, header, fieldspecs, records)

    if print_data_dict:
        parser.write_dd(input_file.name, output_file)


def main(argv=None):
    description = 'Convert CSV to DBF.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', type=argparse.FileType('r', 0), help='input csv', required=True)
    parser.add_argument('--output', type=argparse.FileType('w', 0), help='output dbf file', required=False)
    parser.add_argument('--mapdbf', type=argparse.FileType('r', 0), help='original dbf file used for joining', required=False)
    parser.add_argument('--mapfrom', type=str, help='when mapping, this is the field name matched in the dbf', required=False)
    parser.add_argument('--mapto', type=str, help='when mapping, this is the field name matched in the csv', required=False)
    parser.add_argument('--dd', default=False, required=False, action='store_true', help='output a data dictionary made from the header')

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    _convert_csv_to_dbf(input_file, output_file, args.mapdbf,
                        args.mapfrom, args.mapto,
                        print_data_dict=args.dd)


if __name__ == "__main__":
    sys.exit(main())
