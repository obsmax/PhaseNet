#!/usr/bin/env python
import sys, glob, os
import numpy as np

"""
generate a csv file for phasenet_run.py argument --data_list, from a sds archive
print to stdout
"""

HEADER = 'network,station,location,channel,dataquality,year,julday'
LINEFMT = '{network},{station},{location},{channel},{dataquality},{year},{julday}'

# def rglob(dirname):
#     if os.path.isfile(dirname) or os.path.islink(dirname):
#         yield dirname
#         raise StopIteration
#
#     for item in glob.iglob(os.path.join(dirname, '*')):
#         if os.path.isdir(item):
#             for iitem in rglob(item):
#                 yield iitem
#         else:
#             yield item


if __name__ == '__main__':

    sds = sys.argv[1]  # sds root directory
    assert os.path.isdir(sds)

    lines = []
    searchpath = os.path.join(sds, "[0-9][0-9][0-9][0-9]", "??", "*", "??Z.?")

    for dirname in glob.iglob(searchpath):
        if not os.path.isdir(dirname) and not os.path.islink(dirname):
            continue

        channel, dq = os.path.basename(dirname).split('.')
        dirname = os.path.dirname(dirname)

        station = os.path.basename(dirname)
        dirname = os.path.dirname(dirname)

        network = os.path.basename(dirname)
        dirname = os.path.dirname(dirname)

        year = os.path.basename(dirname)

        filesearch = os.path.join(
            sds, year, network, station,
            f"{channel}.{dq}", f"{network}.{station}.*.{channel}.{dq}.{year}.[0-9][0-9][0-9]")

        for filename in glob.iglob(filesearch):
            if not os.path.isfile(filename):
                continue

            location = filename.split('.')[-5]
            julday = filename.split('.')[-1]

            lines.append((network, station, location, channel.replace('Z', "?"),
                dq, year, julday))

    # convert to arrays
    network, station, location, channel, \
        dataquality, year, julday = \
        [np.array(item, str) for item in zip(*lines)]

    i_sort = np.lexsort((channel, station, network, julday, year))

    print(HEADER)
    for i in i_sort:
        print(f'{network[i]},{station[i]},{location[i]},{channel[i]},{dataquality[i]},{year[i]},{julday[i]}')



