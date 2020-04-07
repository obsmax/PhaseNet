#!/usr/bin/env python
import sys, glob, os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from obspy import read, UTCDateTime


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
                        nargs="+",
                        default=None,
                        help="day to display yyyy.jjj or day range ystart.jstart yend.jend")

    parser.add_argument("--picks_only",
                        action="store_true")

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

    if len(args.day) == 1:
        day = args.day[0]
        year, julday = np.array(day.split('.'), int)
        starttime = UTCDateTime(year, julday=julday)
        endtime = starttime + 24. * 3600.

        year_days = [(year, julday)]

    elif len(args.day) == 2:
        daystart, dayend = args.day
        yearstart, juldaystart = np.array(daystart.split('.'), int)
        yearend, juldayend = np.array(dayend.split('.'), int)
        starttime = UTCDateTime(yearstart, julday=juldaystart)
        endtime = UTCDateTime(yearend, julday=juldayend) + 24. * 3600.
        if not endtime > starttime:
            raise ValueError

        t = starttime + 12. * 3600.
        year_days = []
        while t < endtime:
            year_days.append((t.year, t.julday))
            t += 24. * 3600.

    else:
        raise ValueError(args.day)

    pickdata = pd.read_csv(pickfile, header=0)
    pickdata['time'] = np.asarray([UTCDateTime(_).timestamp for _ in pickdata['time']], float)

    time_selection = (starttime.timestamp <= pickdata['time']) & \
                     (pickdata['time'] <= endtime.timestamp)

    seedids = args.seedid
    input_data = {}

    for year, julday in year_days:
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
                daykey = f'{year}.{julday}'
                component = channel[2]

                # create entries if not exist
                input_data \
                    .setdefault(station_key, {}) \
                    .setdefault(daykey, {}) \
                    .setdefault(component, mseedfile)

                # find rows in pick.csv matching the current station
                seedid_selection = pickdata['seedid'] == station_key
                pick_selection = time_selection & seedid_selection
                if not pick_selection.any():
                    warnings.warn(f'no picks found for {os.path.basename(mseedfile)}')

                input_data[station_key][daykey]["picks"] = {
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
                        input_data[station_key][daykey].setdefault(phasename, predfile)
    return input_data


def display_data(stationkey, daykey, station_day_entry):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for comp in "ENZPS":
        try:
            st = read(station_day_entry[comp], format="MSEED")
        except KeyError:
            continue

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
    for phasename, time, proba in zip(station_day_entry['picks']['phasenames'],
                                      station_day_entry['picks']['times'],
                                      station_day_entry['picks']['probabilities']):
        # TODO use linecollections
        color = {"Z": 'k', "N": "k", "E": "k", "P": "r", "S": "b"}[phasename]
        ax.plot([time, time], ylim,
                color=color,
                alpha=proba)

    ax.set_ylim(ylim)
    ax.set_title(f'{stationkey} {daykey}')
    return fig


def display_data_picks_only(input_data):

    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    ax = fig.add_subplot(111)

    segments = []
    colors = []

    tmin, tmax = np.inf, -np.inf
    ylabels = []
    for nsta, (stationkey, day_entry) in enumerate(input_data.items()):
        for daykey, station_day_entry in day_entry.items():

            for phasename, time, proba in \
                zip(station_day_entry['picks']['phasenames'],
                    station_day_entry['picks']['times'],
                    station_day_entry['picks']['probabilities']):

                segments.append(np.column_stack(([time, time], [nsta - 0.5, nsta + 0.5])))
                colors.append({"P": 'r', 'S': 'b'}[phasename])
                tmin = min([tmin, time])
                tmax = max([tmax, time])
        ylabels.append(stationkey)

    lc = LineCollection(segments, colors=colors)
    ax.add_collection(lc)
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-0.5, nsta + 0.5)
    ax.set_yticks(list(range(nsta+1)))
    ax.set_yticklabels(ylabels, rotation=45, verticalalignment="top", horizontalalignment="right")


def main(args, parser):
    try:
        input_data = find_input_data(args)
        # display_data(mseedfile, predfiles, pickphasenames, picktimes, pickprobas)
        if args.picks_only:
            fig = display_data_picks_only(input_data)
            plt.show()
            plt.close(fig)

        else:
            for stationkey, day_entry in input_data.items():
                for daykey, station_day_entry in day_entry.items():
                    if not len(station_day_entry['picks']['times']):
                        # skip traces with no picks
                        continue
                    fig = display_data(stationkey, daykey, station_day_entry)
                    plt.show()
                    plt.close(fig)

    except (IOError, Exception) as e:
        parser.print_help()
        raise e


if __name__ == '__main__':
    main(*read_args())


