from __future__ import division
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from PhaseNet.data_reader import Config, decode_sample_name
from PhaseNet.detect_peaks import detect_peaks
import logging
from obspy.core.trace import Trace


def detect_peaks_thread(i, pred, fname=None, result_dir=None, args=None):
    input_length = pred.shape[1]
    nedge = input_length // 4  # do not pick maxima in the 25% edge zone each side

    if args is None:
        itp = detect_peaks(pred[i, nedge:-nedge, 0, 1], mph=0.5, mpd=0.5 / Config().dt, show=False)
        its = detect_peaks(pred[i, nedge:-nedge, 0, 2], mph=0.5, mpd=0.5 / Config().dt, show=False)
    else:
        itp = detect_peaks(pred[i, nedge:-nedge, 0, 1], mph=args.tp_prob, mpd=0.5 / Config().dt, show=False)
        its = detect_peaks(pred[i, nedge:-nedge, 0, 2], mph=args.ts_prob, mpd=0.5 / Config().dt, show=False)

    itp = [_ + nedge for _ in itp]
    its = [_ + nedge for _ in its]

    prob_p = pred[i, itp, 0, 1]
    prob_s = pred[i, its, 0, 2]
    if (fname is not None) and (result_dir is not None):
        npzout = os.path.join(result_dir, fname[i].decode())
        pathout = os.path.dirname(npzout)

        os.makedirs(pathout, exist_ok=True)

        np.savez(npzout,
                 pred=pred[i],
                 itp=itp,
                 its=its,
                 prob_p=prob_p,
                 prob_s=prob_s)

    return [(itp, prob_p), (its, prob_s)]


def plot_result_thread(
        i, pred, X, Y=None, itp=None, its=None,
        itp_pred=None, its_pred=None, fname=None, figure_dir=None):

    dt = Config.dt
    t = np.arange(0, pred.shape[1]) * dt
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.05, 0.77]

    plt.figure(i)
    # fig_size = plt.gcf().get_size_inches()
    # plt.gcf().set_size_inches(fig_size*[1, 1.2])

    # ================================
    plt.subplot(411)
    plt.plot(t, X[i, :, 0, 0], 'k', label='E', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(X[i, :, 0, 0])
    tmp_max = np.max(X[i, :, 0, 0])
    if (itp is not None) and (its is not None):
        for j in range(len(itp[i])):
            if j == 0:
                plt.plot([itp[i][j] * dt, itp[i][j] * dt], [tmp_min, tmp_max], 'b', label='P', linewidth=0.5)
            else:
                plt.plot([itp[i][j] * dt, itp[i][j] * dt], [tmp_min, tmp_max], 'b', linewidth=0.5)
        for j in range(len(its[i])):
            if j == 0:
                plt.plot([its[i][j] * dt, its[i][j] * dt], [tmp_min, tmp_max], 'r', label='S', linewidth=0.5)
            else:
                plt.plot([its[i][j] * dt, its[i][j] * dt], [tmp_min, tmp_max], 'r', linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
             transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

    # ================================
    plt.subplot(412)
    plt.plot(t, X[i, :, 0, 1], 'k', label='N', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(X[i, :, 0, 1])
    tmp_max = np.max(X[i, :, 0, 1])
    if (itp is not None) and (its is not None):
        for j in range(len(itp[i])):
            plt.plot([itp[i][j] * dt, itp[i][j] * dt], [tmp_min, tmp_max], 'b', linewidth=0.5)
        for j in range(len(its[i])):
            plt.plot([its[i][j] * dt, its[i][j] * dt], [tmp_min, tmp_max], 'r', linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
             transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

    # ================================
    plt.subplot(413)
    plt.plot(t, X[i, :, 0, 2], 'k', label='Z', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(X[i, :, 0, 2])
    tmp_max = np.max(X[i, :, 0, 2])
    if (itp is not None) and (its is not None):
        for j in range(len(itp[i])):
            plt.plot([itp[i][j] * dt, itp[i][j] * dt], [tmp_min, tmp_max], 'b', linewidth=0.5)
        for j in range(len(its[i])):
            plt.plot([its[i][j] * dt, its[i][j] * dt], [tmp_min, tmp_max], 'r', linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
             transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

    # ================================
    plt.subplot(414)
    if Y is not None:
        plt.plot(t, Y[i, :, 0, 1], 'b', label='P', linewidth=0.5)
        plt.plot(t, Y[i, :, 0, 2], 'r', label='S', linewidth=0.5)
    plt.plot(t, pred[i, :, 0, 1], '--g', label='$\hat{P}$', linewidth=0.5)
    plt.plot(t, pred[i, :, 0, 2], '-.m', label='$\hat{S}$', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    if (itp_pred is not None) and (its_pred is not None):
        for j in range(len(itp_pred)):
            plt.plot([itp_pred[j] * dt, itp_pred[j] * dt], [-0.1, 1.1], '--g', linewidth=0.5)
        for j in range(len(its_pred)):
            plt.plot([its_pred[j] * dt, its_pred[j] * dt], [-0.1, 1.1], '-.m', linewidth=0.5)
    plt.ylim([-0.05, 1.05])
    plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
             transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.legend(loc='upper right', fontsize='small')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')

    plt.tight_layout()
    plt.gcf().align_labels()

    try:
        plt.savefig(os.path.join(figure_dir,
                                 fname[i].decode().replace('.npz', "") + '.png'),
                    bbox_inches='tight')
    except FileNotFoundError:
        # if not os.path.exists(os.path.dirname(os.path.join(figure_dir, fname[i].decode()))):
        os.makedirs(os.path.dirname(os.path.join(figure_dir, fname[i].decode())), exist_ok=True)
        plt.savefig(os.path.join(figure_dir,
                                 fname[i].decode().replace('.npz', "") + '.png'),
                    bbox_inches='tight')
    # plt.savefig(os.path.join(figure_dir,
    #            fname[i].decode().split('/')[-1].rstrip('.npz')+'.png'),
    #            bbox_inches='tight')
    # plt.savefig(os.path.join(figure_dir,
    #             fname[i].decode().split('/')[-1].rstrip('.npz')+'.pdf'),
    #             bbox_inches='tight')
    plt.close(i)
    return 0


def save_predictions_to_hdf5_archive(hdf5_pointer, fname_batch, pred_batch):
    """
    :param hdf5_pointer:
    :param fname_batch:
    :param pred_batch:
    """

    # location of the sample results inside the hdf5 archive
    HDF5PATH = "{year:04d}/{network:s}/{station:s}/{channel2:2s}{phasename:1s}.{dataquality:1s}/{julday}"

    for i in range(len(fname_batch)):
        seedid, sample_start, sampling_rate, sample_npts, \
        (network, station, location, channel2, dataquality) = \
            decode_sample_name(sample_name=fname_batch[i].decode())

        midtime = sample_start + 0.5 * (sample_npts - 1) / sampling_rate
        for nphase, phasename in enumerate("PS"):
            groupname = HDF5PATH.format(
                year=midtime.year,
                julday=midtime.julday,
                network=network, station=station,
                channel2=channel2, phasename=phasename,
                dataquality=dataquality)

            # AVOID THE EDGES OF THE SAMPLE (because of the 50% overlap)
            n = sample_npts // 4

            try:
                grp = hdf5_pointer[groupname]
            except KeyError:
                grp = hdf5_pointer.create_group(groupname)

            sample_dataset = grp.create_dataset(
                fname_batch[i].decode(),
                data=255 * pred_batch[i, n:-n, 0, nphase + 1],  # 1 for P and 2 for S
                dtype=np.dtype('uint8'))  # to save disk space, proba scaled by 255

            sample_dataset.attrs["network"] = network
            sample_dataset.attrs["station"] = station
            sample_dataset.attrs["location"] = location
            sample_dataset.attrs["channel"] = channel2 + phasename
            sample_dataset.attrs["dataquality"] = dataquality
            sample_dataset.attrs["starttime"] = str(sample_start + n / sampling_rate)
            sample_dataset.attrs["sampling_rate"] = sampling_rate


def reform_mseed_files_from_predictions(hdf5_pointer, result_dir):
    MSEEDFILE = \
        os.path.join("{result_dir}", "{year}", "{network}", "{station}", "{channeldq}",
                     "{network}.{station}.{location}.{channeldq}.{year}.{julday}")

    from obspy.core import Stream, Trace, UTCDateTime
    for year, ygrp in hdf5_pointer.items():
        for network, ngrp in ygrp.items():
            for station, sgrp in ngrp.items():
                for channeldq, cgrp in sgrp.items():
                    for julday, jgrp in cgrp.items():
                        # group sample predictions per julian day
                        stream = Stream([])
                        trace = None
                        for sample_dataset_name, sample_dataset in jgrp.items():
                            starttime = UTCDateTime(sample_dataset.attrs["starttime"])

                            trace = Trace(
                                #data=sample_dataset[:],
                                data=np.asarray(sample_dataset[:], np.dtype('int32')),
                                header={
                                    "network":sample_dataset.attrs['network'],
                                    "station": sample_dataset.attrs['station'],
                                    "location": sample_dataset.attrs['location'],
                                    "channel": sample_dataset.attrs['channel'],
                                    "starttime": starttime,
                                    "sampling_rate": sample_dataset.attrs["sampling_rate"],
                                    "mseed": {"dataquality": sample_dataset.attrs["dataquality"]}})

                            stream.append(trace)
                        if not len(stream):
                            continue

                        mseedfile = MSEEDFILE.format(
                            result_dir=result_dir, year=year, julday=julday,
                            network=network, station=station,
                            location=trace.stats.location,
                            channeldq=channeldq)
                        print(mseedfile)

                        os.makedirs(os.path.dirname(mseedfile), exist_ok=True)
                        stream.merge(fill_value=0, interpolation_samples=0)
                        stream[0].trim(
                            UTCDateTime(int(year), julday=int(julday), hour=0),
                            UTCDateTime(int(year), julday=int(julday), hour=0) + 24. * 3600.)
                        stream.write(mseedfile, format="MSEED")

# def extract_preds_sds_thread(
#         i, pred, fname=None, result_dir=None):
#
#     PREDPATH = os.path.join(
#         "{result_dir}",
#         "{year}", "{network}", "{station}", "{channel2}{phasename}.{dataquality}")
#
#     PREDFILE = "{network}.{station}.{location}.{channel2}{phasename}.{dataquality}." \
#                "{year:04d}.{julday:03d}.{hour:02d}.{minute:02d}.{second:09.6f}.mseed"
#
#     if (fname is not None) and (result_dir is not None):
#         seedid, batch_start, sampling_rate, \
#             (network, station, location, channel2, dataquality) = \
#             decode_sample_name(sample_name=fname[i].decode())
#
#         for nphase, phasename in enumerate("PS"):
#             pathout = PREDPATH.format(
#                 result_dir=result_dir,
#                 year=batch_start.year,
#                 julday=batch_start.julday,
#                 network=network,
#                 station=station,
#                 location=location,
#                 channel2=channel2,
#                 phasename=phasename,
#                 dataquality=dataquality)
#
#             if not os.path.isdir(pathout):
#                 print(f'creating : {pathout}')
#                 os.makedirs(pathout, exist_ok=True)
#
#             fileout = PREDFILE.format(
#                 year=batch_start.year,
#                 julday=batch_start.julday,
#                 hour=batch_start.hour,
#                 minute=batch_start.minute,
#                 second=batch_start.second + 1.e-6 * batch_start.microsecond,
#                 network=network,
#                 station=station,
#                 location=location,
#                 channel2=channel2,
#                 phasename=phasename,
#                 dataquality=dataquality,
#                 i=i)
#
#             trace = Trace(
#                 data=pred[i, :, 0, nphase+1],
#                 header={
#                     "network": network,
#                     "station": station,
#                     "location": location,
#                     "channel": channel2 + phasename,
#                     "starttime": batch_start,
#                     "sampling_rate": sampling_rate})
#
#             trace.write(
#                 os.path.join(pathout, fileout), format="MSEED")


def postprocessing_thread(i, pred, X, Y=None, itp=None, its=None, fname=None, result_dir=None, figure_dir=None,
                          args=None):
    """
    :param i: batch number
    :param pred: probability functions for P and S phases and?
    :param X: seismic data
    :param Y:
    :param itp: indexs of picked P phases
    :param its: indexs of picked S phases
    :param fname:
    :param result_dir:
    :param figure_dir:
    """
    (itp_pred, prob_p), (its_pred, prob_s) = \
        detect_peaks_thread(i, pred, fname, result_dir, args)

    if (fname is not None) and (figure_dir is not None):
        plot_result_thread(i, pred, X, Y, itp, its, itp_pred, its_pred, fname, figure_dir)
    return [(itp_pred, prob_p), (its_pred, prob_s)]


def clean_queue(picks):
    clean = []
    for i in range(len(picks)):
        tmp = []
        for j in picks[i]:
            if j != 0:
                tmp.append(j)
        clean.append(tmp)
    return clean


def clean_queue_thread(picks):
    tmp = []
    for j in picks:
        if j != 0:
            tmp.append(j)
    return tmp


def metrics(TP, nP, nT):
    '''
    TP: true positive
    nP: number of positive picks
    nT: number of true picks
    '''
    precision = TP / nP
    recall = TP / nT
    F1 = 2 * precision * recall / (precision + recall)
    return [precision, recall, F1]


def correct_picks(picks, true_p, true_s, tol):
    dt = Config().dt
    if len(true_p) != len(true_s):
        print("The length of true P and S pickers are not the same")
    num = len(true_p)
    TP_p = 0
    TP_s = 0
    nP_p = 0
    nP_s = 0
    nT_p = 0
    nT_s = 0
    diff_p = []
    diff_s = []
    for i in range(num):
        nT_p += len(true_p[i])
        nT_s += len(true_s[i])
        nP_p += len(picks[i][0][0])
        nP_s += len(picks[i][1][0])

        if len(true_p[i]) > 1 or len(true_s[i]) > 1:
            print(i, picks[i], true_p[i], true_s[i])
        tmp_p = np.array(picks[i][0][0]) - np.array(true_p[i])[:, np.newaxis]
        tmp_s = np.array(picks[i][1][0]) - np.array(true_s[i])[:, np.newaxis]
        TP_p += np.sum(np.abs(tmp_p) < tol / dt)
        TP_s += np.sum(np.abs(tmp_s) < tol / dt)
        diff_p.append(tmp_p[np.abs(tmp_p) < 0.5 / dt])
        diff_s.append(tmp_s[np.abs(tmp_s) < 0.5 / dt])

    return [TP_p, TP_s, nP_p, nP_s, nT_p, nT_s, diff_p, diff_s]


def calculate_metrics(picks, itp, its, tol=0.1):
    TP_p, TP_s, nP_p, nP_s, nT_p, nT_s, diff_p, diff_s = correct_picks(picks, itp, its, tol)
    precision_p, recall_p, f1_p = metrics(TP_p, nP_p, nT_p)
    precision_s, recall_s, f1_s = metrics(TP_s, nP_s, nT_s)

    logging.info("Total records: {}".format(len(picks)))
    logging.info("P-phase:")
    logging.info("True={}, Predict={}, TruePositive={}".format(nT_p, nP_p, TP_p))
    logging.info("Precision={:.3f}, Recall={:.3f}, F1={:.3f}".format(precision_p, recall_p, f1_p))
    logging.info("S-phase:")
    logging.info("True={}, Predict={}, TruePositive={}".format(nT_s, nP_s, TP_s))
    logging.info("Precision={:.3f}, Recall={:.3f}, F1={:.3f}".format(precision_s, recall_s, f1_s))
    return [precision_p, recall_p, f1_p], [precision_s, recall_s, f1_s]
