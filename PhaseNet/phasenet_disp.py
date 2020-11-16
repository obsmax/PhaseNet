#!/usr/bin/env python
import sys, glob, os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.dates import date2num, DateFormatter
from datetime import datetime
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

    parser.add_argument("--mode",
                        default=None,
                        help="picks-only/all")

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

    parser.add_argument("--time",
                        nargs="+",
                        default=None,
                        help="time range at format ystart.jstart.hourstart.minutestart yend.jend.hourend.minuteend")

    parser.add_argument("--decim",
                        default=5000,
                        type=int,
                        help="number of windows for the (obspy-like) decimation for display or 0")

    if len(sys.argv) == 1:
        # print help if no arguments passed
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args, parser


def decode_time_args(args):
    def decode_arg(arg):
        year, julday, hour, minute = np.array(arg.split('.'), int)
        return UTCDateTime(year, julday=julday, hour=hour, minute=minute)

    start = decode_arg(args.time[0])
    end = decode_arg(args.time[1])

    if not start < end:
        raise ValueError(start, end)

    if not start.year == end.year and start.julday == end.julday:
        raise NotImplementedError('start and end must point to the same day')

    return start, end


def utcdatetime2datetime(utcdatetime_or_timestamp):
    return date2num(datetime.utcfromtimestamp(float(utcdatetime_or_timestamp)))


def utcdatetimes2datetimes(utcdatetimes_or_timestamps):
    return np.asarray([utcdatetime2datetime(_) for _ in utcdatetimes_or_timestamps], float)


def data_snapshot(data_array, delta, nwin=256):
    """an obspy-like data aliaser for display"""
    npts = len(data_array)
    t = np.arange(npts) * delta
    if npts <= nwin:
        data_points, data_values = t, data_array
        return data_points, data_values
    n_per_win = int(np.ceil(npts / float(nwin)))
    npad = n_per_win * nwin - npts
    lwin = n_per_win * delta
    assert npts + npad == n_per_win * nwin
    d = np.concatenate((data_array,
                        data_array[-1] * np.ones(npad, data_array.dtype)))
    d = d.reshape((nwin, n_per_win))
    min_values = d.min(axis=1)
    max_values = d.max(axis=1)

    data_values = np.zeros(2 * nwin + 2, d.dtype)
    data_points = np.zeros(2 * nwin + 2, t.dtype)

    data_values[1:-1:2] = max_values
    data_values[2::2] = min_values
    data_values[0] = data_array[0]
    data_values[-1] = data_array[-1]
    data_points[0] = t[0]
    data_points[-1] = t[-1]
    data_points[1:-1:2] = data_points[2::2] = np.arange(nwin) * lwin + 0.5 * lwin

    return data_points, data_values


def find_input_data(args):
    """
    load data and place it in a dictionary
    input_data
        [network.station.location.channel[:2].dataquality]
            [year.julday]
                [phasename] -> name of a mseed file (phasename=E,N,Z,P or S)
                [picks]
                    [times]        -> array of UTCDateTimes with the picks
                    [phasnames]    -> array of phasenames (P or S)
                    [probabilities]-> array of probabilitues
    """

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise IOError(f"{data_dir} not found")

    output_dir = args.output_dir
    if not os.path.isdir(data_dir):
        raise IOError(f"{output_dir} not found")

    pickfile = os.path.join(output_dir, "picks.csv")
    if not os.path.isfile(pickfile):
        raise IOError(f'{pickfile} not found')

    starttime, endtime = decode_time_args(args)

    pickdata = pd.read_csv(pickfile, header=0)
    pickdata['time'] = np.asarray([UTCDateTime(_).timestamp for _ in pickdata['time']], float)

    time_selection = (starttime.timestamp <= pickdata['time']) & \
                     (pickdata['time'] <= endtime.timestamp)
    if not time_selection.any():
        raise ValueError(f'no input data found for {args}')

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
            year=starttime.year, julday=starttime.julday)

        mseedfiles = glob.glob(mseed_search_path)
        if not len(mseedfiles):
            # warnings.warn(f'no mseed file found for {mseed_search_path}')
            continue

        for mseedfile in mseedfiles:

            # seedid items, no more wildcards
            network, station, location, channel, dataquality = \
                os.path.basename(mseedfile).split('.')[:5]

            station_key = f"{network}.{station}.{location}.{channel[:2]}.{dataquality}"
            daykey = f'{starttime.year}.{starttime.julday}'
            component = channel[2]

            # create entries if not exist
            input_data \
                .setdefault(station_key, {}) \
                .setdefault(daykey, {}) \
                .setdefault(component, mseedfile)

            # find rows in pick.csv matching the current station
            seedid_selection = pickdata['seedid'] == station_key
            pick_selection = time_selection & seedid_selection

            input_data[station_key][daykey]["picks"] = {
                'phasenames': pickdata["phasename"][pick_selection],
                'times': pickdata["time"][pick_selection],
                'probabilities': pickdata["probability"][pick_selection]}

            if not pick_selection.any():
                # warnings.warn(f'no picks found for {os.path.basename(mseedfile)}')
                pass #continue

            # find prediction traces if any
            for phasename in "PS":
                pred_search_path = SDSPATH.format(
                    data_dir=os.path.join(output_dir, "results"),
                    network=network, station=station,
                    location=location,
                    channel=channel[:2] + phasename,
                    dataquality=dataquality,
                    year=starttime.year, julday=starttime.julday)

                predfiles = glob.glob(pred_search_path)
                if not len(predfiles):
                    #warnings.warn(f'no prediction file found for {pred_search_path}')
                    pass

                for predfile in predfiles:
                    input_data[station_key][daykey].setdefault(phasename, predfile)
    return input_data


def display_data_picks_only(input_data):

    fig = plt.figure(figsize=(9, 4))
    fig.subplots_adjust(left=0.2, bottom=0.2)
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

                time = utcdatetime2datetime(time)

                segments.append(np.column_stack(([time, time],
                                                 [nsta - 0.5, nsta + 0.5])))
                colors.append({"P": 'r', 'S': 'b'}[phasename])
                tmin = min([tmin, time])
                tmax = max([tmax, time])
        ylabels.append(stationkey)

    if np.isinf(tmin):
        plt.close(fig)
        return

    lc = LineCollection(segments, colors=colors)
    ax.add_collection(lc)
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-0.5, nsta + 0.5)
    ax.set_yticks(list(range(nsta+1)))
    ax.set_yticklabels(ylabels, rotation=45, verticalalignment="top", horizontalalignment="right")

    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")


def display_data_all(input_data, starttime, endtime, gain=0.01,
                     pred_decim=20.,
                     use_obspy_aliaser=True,
                     obspy_aliaser_nwin=5000):
    #assert endtime - starttime <= 24 * 3600.
    daykey = f"{starttime.year}.{starttime.julday}"  # assume same day for endtime

    fig = plt.figure(figsize=(9, 4))
    fig.subplots_adjust(left=0.2, bottom=0.2)
    ax = fig.add_subplot(111)

    data_segments, data_colors = [], []
    pred_segments, pred_colors = [], []
    pick_segments, pick_colors = [], []
    color_code = {"Z": "k", "N": "k", "E": "k", "P": "r", "S": "b"}

    offset = -1
    yticks = []
    for stationkey, stationentry in input_data.items():

        try:
            input_data[stationkey][daykey]["Z"]
            input_data[stationkey][daykey]["N"]
            input_data[stationkey][daykey]["E"]
            input_data[stationkey][daykey]["P"]
            input_data[stationkey][daykey]["S"]
        except KeyError:
            continue

        offset += 1
        yticks.append(stationkey)

        # display waveforms
        for ncomp, comp in enumerate("ZNE"):
            mseedfile = input_data[stationkey][daykey][comp]

            st = read(mseedfile, format="MSEED",
                      starttime=starttime, endtime=endtime)
            for tr in st:
                tr.detrend('constant')

            st.merge(fill_value=0)
            trace = st[0]


            trace.data /= trace.data.std()

            # display aliased data !!
            if use_obspy_aliaser:
                trace_time, trace_data = \
                    data_snapshot(data_array=trace.data, delta=trace.stats.delta, nwin=obspy_aliaser_nwin)
                trace_time += trace.stats.starttime.timestamp

            else:
                data_decim = 50.
                trace.resample(sampling_rate=data_decim, no_filter=True, strict_length=False)

                trace_time = trace.stats.starttime.timestamp + \
                             np.arange(trace.stats.npts) * trace.stats.delta
                trace_data = trace.data
            print(trace)

            trace_data = gain * trace_data + offset + ncomp / 5. - 1./5.

            trace_time = utcdatetimes2datetimes(trace_time)

            data_segments.append(np.column_stack((trace_time, trace_data)))
            data_colors.append(color_code[comp])

        # display prediction series
        for ncomp, comp in enumerate("PS"):
            try:
                mseedfile = input_data[stationkey][daykey][comp]
            except KeyError:
                continue

            st = read(mseedfile, format="MSEED",
                      starttime=starttime, endtime=endtime)
            st.merge(fill_value=0)
            trace = st[0]

            # display aliased data !!
            trace.resample(sampling_rate=pred_decim, no_filter=True, strict_length=False)
            print(trace)
            trace.data = np.asarray(trace.data, float) / 255.  # by convention
            trace.data[trace.data <= 0.001] = np.nan

            trace_time = trace.stats.starttime.timestamp + \
                         np.arange(trace.stats.npts) * trace.stats.delta

            trace_data = 0.5 * trace.data + offset #+ ncomp / 2. - 1. / 4.

            trace_time = utcdatetimes2datetimes(trace_time)

            pred_segments.append(np.column_stack((trace_time, trace_data)))
            pred_colors.append(color_code[comp])

        # display phase picks
        picks = input_data[stationkey][daykey]["picks"]
        for time, phase, proba in zip(picks['times'], picks['phasenames'], picks['probabilities']):

            time = utcdatetime2datetime(time)

            pick_segments.append(np.column_stack(([time, time], [offset-0.5, offset+0.5])))
            pick_colors.append(color_code[phase])

    lc_data = LineCollection(data_segments, colors=data_colors, alpha=0.3)
    lc_pred = LineCollection(pred_segments, colors=pred_colors, alpha=0.8)
    lc_pick = LineCollection(pick_segments, colors=pick_colors, alpha=0.8)

    ax.add_collection(lc_data)
    ax.add_collection(lc_pred)
    ax.add_collection(lc_pick)
    ax.set_yticks(list(range(offset+1)))
    ax.set_yticklabels(yticks, rotation=45., horizontalalignment="right", verticalalignment="top")

    ax.set_xlim(utcdatetimes2datetimes([starttime.timestamp, endtime.timestamp]))
    ax.set_ylim(-0.5, offset + 1.0)

    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    return fig


def main(args, parser):
    try:
        input_data = find_input_data(args)

        if args.mode == "picks-only":
            fig = display_data_picks_only(input_data)
            plt.show()
            plt.close(fig)

        elif args.mode == "all":
            starttime, endtime = decode_time_args(args)

            use_obspy_aliaser = args.decim > 0
            obspy_aliaser_nwin = args.decim

            fig = display_data_all(
                input_data, starttime, endtime,
                use_obspy_aliaser=use_obspy_aliaser,
                obspy_aliaser_nwin=obspy_aliaser_nwin)

            plt.show()
            plt.close(fig)

        else:
            raise NotImplementedError(f'unknown mode {args.mode}')

    except (IOError, Exception) as e:
        parser.print_help()
        raise e


if __name__ == '__main__':
    main(*read_args())


