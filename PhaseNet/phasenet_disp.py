#!/usr/bin/env python
import sys, glob, os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime


# conventional path name for SDS data archive
SDSPATH = os.path.join(
    "{data_dir}", "{year}",
    "{network}", "{station}",
    "{channel}.{dataquality}",
    "{network}.{station}.{location}.{channel}.{dataquality}"
    ".{year:04d}.{julday:03d}")


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        help="Root of the sds tree")

    parser.add_argument("--output_dir",
                        default=None,
                        help="Output directory where results are stored")

    parser.add_argument("--seedid",
                        nargs="+",
                        default=None,
                        help="stations to display, wildcards allowed with quotes, "
                             "network.station.location.channel.dataquality")

    parser.add_argument("--day",
                        default=None,
                        help="day to display yyyy.jjj")

    if len(sys.argv) == 1:
        # print help if no arguments passed
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args, parser


def find_input_data(args):

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise IOError(f"{data_dir} not found")

    output_dir = args.output_dir
    if not os.path.isdir(data_dir):
        raise IOError(f"{output_dir} not found")

    pickfile = os.path.join(output_dir, "picks.csv")
    if not os.path.isfile(pickfile):
        raise IOError(f'{pickfile} not found')

    day = args.day
    year, julday = np.array(day.split('.'), int)
    starttime = UTCDateTime(year, julday=julday)
    endtime = starttime + 24. * 3600.

    pickdata = pd.read_csv(pickfile, header=0)
    pickdata['time'] = np.asarray([UTCDateTime(_).timestamp for _ in pickdata['time']], float)
    # pickdata['network'], pickdata['station'], pickdata['location'], \
    #     pickdata['channel2'], pickdata['dataquality'] = \
    #     [np.array(_, str) for _ in zip(*[_.split('.') for _ in pickdata['seedid']])]

    time_selection = (starttime.timestamp <= pickdata['time']) & \
                     (pickdata['time'] <= endtime.timestamp)

    seedids = args.seedid
    input_data = {}

    for seedid in seedids:
        # seedid may include wildcards
        network, station, location, channel, dataquality =\
            seedid.split('.')

        mseed_search_path = SDSPATH.format(
            data_dir=data_dir,
            network=network, station=station,
            location=location, channel=channel,
            dataquality=dataquality,
            year=year, julday=julday)

        mseedfiles = glob.glob(mseed_search_path)
        if not len(mseedfiles):
            warnings.warn(f'no mseed file found for {mseed_search_path}')
            continue

        for mseedfile in mseedfiles:

            # seedid items, no more wildcards
            network, station, location, channel, dataquality = \
                os.path.basename(mseedfile).split('.')[:5]

            station_key = f"{network}.{station}.{location}.{channel[:2]}.{dataquality}"
            component = channel[2]

            #
            input_data.setdefault(station_key, {}).setdefault(component, mseedfile)

            # find rows in pick.csv matching the current station
            seedid_selection = pickdata['seedid'] == station_key
            pick_selection = time_selection & seedid_selection
            if not pick_selection.any():
                warnings.warn(f'no picks found for {os.path.basename(mseedfile)}')

            input_data[station_key]["picks"] = {
                'phasenames': pickdata["phasename"][pick_selection],
                'times': pickdata["time"][pick_selection],
                'probabilities': pickdata["probability"][pick_selection]}

            # find prediction traces if any
            for phasename in "PS":
                pred_search_path = SDSPATH.format(
                    data_dir=os.path.join(output_dir, "results"),
                    network=network, station=station,
                    location=location,
                    channel=channel[:2] + phasename,
                    dataquality=dataquality,
                    year=year, julday=julday)

                predfiles = glob.glob(pred_search_path)
                if not len(predfiles):
                    warnings.warn(f'no prediction file found for {pred_search_path}')

                for predfile in predfiles:
                    input_data[station_key].setdefault(phasename, predfile)
    return input_data


def display_data(stationkey, station_entry):
    import obspy
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for comp in "ENZPS":
        st = obspy.read(station_entry[comp], format="MSEED")
        if comp in 'ENZ':
            for tr in st:
                tr.detrend('constant')
            st.merge(fill_value=0.)
            st[0].data *= 0.1 / st[0].data.std()
        elif comp in "PS":
            st[0].data = np.asarray(st[0].data, float) * 3.0 / 255.0

        timearray = st[0].stats.starttime.timestamp + np.arange(st[0].stats.npts) * st[0].stats.delta
        color = {"Z": 'k', "N": "k", "E": "k", "P": "r", "S": "b"}[comp]
        offset = {"Z": 0.0, "N": 1.0, "E": 2.0, "P": 0., "S": 0.}[comp]
        alpha = {"Z": 0.3, "N": 0.3, "E": 0.3, "P": 1.0, "S": 1.0}[comp]
        ax.plot(timearray,
                st[0].data + offset,
                color=color,
                alpha=alpha)

    ylim = [-0.5, 3.5]
    for phasename, time, proba in zip(station_entry['picks']['phasenames'],
                                      station_entry['picks']['times'],
                                      station_entry['picks']['probabilities']):
        color = {"Z": 'k', "N": "k", "E": "k", "P": "r", "S": "b"}[phasename]
        ax.plot([time, time], ylim,
                color=color,
                alpha=proba)

    ax.set_ylim(ylim)
    ax.set_title(stationkey)
    return fig


def main(args, parser):
    try:
        input_data = find_input_data(args)
        # display_data(mseedfile, predfiles, pickphasenames, picktimes, pickprobas)
        for stationkey, station_entry in input_data.items():

            fig = display_data(stationkey, station_entry)
            plt.show()
            plt.close(fig)

    except (IOError, Exception) as e:
        parser.print_help()
        raise e


if __name__ == '__main__':
    main(*read_args())

