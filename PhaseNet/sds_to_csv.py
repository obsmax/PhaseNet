#!/usr/bin/env python
import sys, glob, os
import numpy as np

"""
generate a csv file for phasenet_run.py argument --data_list, from a sds archive
print to stdout
"""


def rglob(dirname):
    if os.path.isfile(dirname) or os.path.islink(dirname):
        yield dirname
        raise StopIteration

    for item in glob.iglob(os.path.join(dirname, '*')):
        if os.path.isdir(item):
            for iitem in rglob(item):
                yield iitem
        else:
            yield item


if __name__ == '__main__':

    sds = sys.argv[1]  # sds root directory
    assert os.path.isdir(sds)

    HEADER = 'network,station,location,channel,dataquality,year,julday'
    LINEFMT = '{network},{station},{location},{channel},{dataquality},{year},{julday}'
    lines = []
    for item in rglob(sds):
        mseedfile = os.path.basename(item)
        try:
            network, station, location, channel, \
            dataquality, year, julday = mseedfile.split('.')
        except ValueError as e:
            if "not enough values to unpack (expected 7," in str(e):
                continue
            else:
                raise e

        if channel.endswith('Z'):
            lines.append((network, station, location, channel.replace('Z', "?"),
                dataquality, year, julday))

    # convert to arrays
    network, station, location, channel, \
        dataquality, year, julday = \
        [np.array(item, str) for item in zip(*lines)]

    i_sort = np.lexsort((channel, station, network, julday, year))

    print(HEADER)
    for i in i_sort:
        print(f'{network[i]},{station[i]},{location[i]},{channel[i]},{dataquality[i]},{year[i]},{julday[i]}')



