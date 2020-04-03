from __future__ import division
import os
import threading
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import scipy.interpolate

pd.options.mode.chained_assignment = None
import obspy
from obspy.core import UTCDateTime
from tqdm import tqdm


SDSPATH = os.path.join(
    "{data_dir}", "{year}",
    "{network}", "{station}",
    "{channel}.{dataquality}",
    "{network}.{station}.{location}.{channel}.{year:.04f}.{julday:.03f}")


class Config(object):
    seed = 100
    use_seed = False
    n_channel = 3
    n_class = 3
    num_repeat_noise = 1
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    X_shape = [3000, 1, n_channel]
    Y_shape = [3000, 1, n_class]
    min_event_gap = 3 * sampling_rate


class DataReader(object):

    def __init__(self,
                 data_dir,
                 data_list,
                 mask_window,
                 queue_size,
                 coord,
                 config=Config()):
        self.config = config
        tmp_list = pd.read_csv(data_list, header=0)
        self.data_list = tmp_list
        self.num_data = len(self.data_list)
        self.data_dir = data_dir
        self.queue_size = queue_size
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.min_event_gap = config.min_event_gap
        self.mask_window = int(mask_window * config.sampling_rate)
        self.coord = coord
        self.threads = []
        self.buffer = {}
        self.buffer_channels = {}
        self.add_placeholder()

    def add_placeholder(self):
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=self.config.X_shape)
        self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=self.config.Y_shape)
        self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                         ['float32', 'float32'],
                                         shapes=[self.config.X_shape, self.config.Y_shape])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.target_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def normalize(self, data):
        data -= np.mean(data, axis=0, keepdims=True)
        std_data = np.std(data, axis=0, keepdims=True)
        assert (std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
        return data

    def adjust_missingchannels(self, data):
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert (tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
            data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def thread_main(self, sess, n_threads=1, start=0):
        stop = False
        while not stop:
            index = list(range(start, self.num_data, n_threads))
            np.random.shuffle(index)
            for i in index:
                fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
                try:
                    if fname not in self.buffer:
                        meta = np.load(fname)
                        self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'],
                                              'channels': meta['channels']}
                    meta = self.buffer[fname]
                except:
                    logging.error("Failed reading {}".format(fname))
                    continue

                channels = meta['channels'].tolist()
                start_tp = meta['itp'].tolist()

                if self.coord.should_stop():
                    stop = True
                    break

                sample = np.zeros(self.X_shape)
                if np.random.random() < 0.95:
                    data = np.copy(meta['data'])
                    itp = meta['itp']
                    its = meta['its']
                    start_tp = itp

                    shift = np.random.randint(-(self.X_shape[0] - self.mask_window),
                                              min([its - start_tp, self.X_shape[0]]) - self.mask_window)
                    sample[:, :, :] = data[start_tp + shift:start_tp + self.X_shape[0] + shift, np.newaxis, :]
                    itp_list = [itp - start_tp - shift]
                    its_list = [its - start_tp - shift]
                else:
                    sample[:, :, :] = np.copy(meta['data'][start_tp - self.X_shape[0]:start_tp, np.newaxis, :])
                    itp_list = []
                    its_list = []

                sample = self.normalize(sample)
                sample = self.adjust_missingchannels(sample)

                if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
                    continue

                target = np.zeros(self.Y_shape)
                for itp, its in zip(itp_list, its_list):
                    if (itp >= target.shape[0]) or (itp < 0):
                        pass
                    elif (itp - self.mask_window // 2 >= 0) and (itp - self.mask_window // 2 < target.shape[0]):
                        target[itp - self.mask_window // 2:itp + self.mask_window // 2, 0, 1] = \
                            np.exp(-(np.arange(-self.mask_window // 2, self.mask_window // 2)) ** 2 / (
                                        2 * (self.mask_window // 4) ** 2))[
                            :target.shape[0] - (itp - self.mask_window // 2)]
                    elif (itp - self.mask_window // 2 < target.shape[0]):
                        target[0:itp + self.mask_window // 2, 0, 1] = \
                            np.exp(-(np.arange(0, itp + self.mask_window // 2) - itp) ** 2 / (
                                        2 * (self.mask_window // 4) ** 2))[
                            :target.shape[0] - (itp - self.mask_window // 2)]
                    if (its >= target.shape[0]) or (its < 0):
                        pass
                    elif (its - self.mask_window // 2 >= 0) and (its - self.mask_window // 2 < target.shape[0]):
                        target[its - self.mask_window // 2:its + self.mask_window // 2, 0, 2] = \
                            np.exp(-(np.arange(-self.mask_window // 2, self.mask_window // 2)) ** 2 / (
                                        2 * (self.mask_window // 4) ** 2))[
                            :target.shape[0] - (its - self.mask_window // 2)]
                    elif (its - self.mask_window // 2 < target.shape[0]):
                        target[0:its + self.mask_window // 2, 0, 2] = \
                            np.exp(-(np.arange(0, its + self.mask_window // 2) - its) ** 2 / (
                                        2 * (self.mask_window // 4) ** 2))[
                            :target.shape[0] - (its - self.mask_window // 2)]
                target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

                sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                                  self.target_placeholder: target})
        return 0

    def start_threads(self, sess, n_threads=8):
        for i in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess, n_threads, i))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self.threads


class DataReader_test(DataReader):

    def add_placeholder(self):
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.itp_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.its_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                         ['float32', 'float32', 'string', 'int32', 'int32'],
                                         shapes=[self.config.X_shape, self.config.Y_shape, [], [None], [None]])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.target_placeholder,
                                           self.fname_placeholder,
                                           self.itp_placeholder, self.its_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_up_to(num_elements)
        return output

    def thread_main(self, sess, n_threads=1, start=0):
        index = list(range(start, self.num_data, n_threads))
        for i in index:
            fname = self.data_list.iloc[i]['fname']
            fp = os.path.join(self.data_dir, fname)
            try:
                if fp not in self.buffer:
                    meta = np.load(fp)
                    self.buffer[fp] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'],
                                       'channels': meta['channels']}
                meta = self.buffer[fp]
            except:
                logging.error("Failed reading {}".format(fp))
                continue

            channels = meta['channels'].tolist()
            start_tp = meta['itp'].tolist()

            if self.coord.should_stop():
                break

            sample = np.zeros(self.X_shape)

            np.random.seed(self.config.seed + i)
            shift = np.random.randint(-(self.X_shape[0] - self.mask_window),
                                      min([meta['its'].tolist() - start_tp, self.X_shape[0]]) - self.mask_window)
            sample[:, :, :] = np.copy(meta['data'][start_tp + shift:start_tp + self.X_shape[0] + shift, np.newaxis, :])
            itp_list = [meta['itp'].tolist() - start_tp - shift]
            its_list = [meta['its'].tolist() - start_tp - shift]

            sample = self.normalize(sample)
            sample = self.adjust_missingchannels(sample)

            if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
                continue

            target = np.zeros(self.Y_shape)
            itp_true = []
            its_true = []
            for itp, its in zip(itp_list, its_list):
                if (itp >= target.shape[0]) or (itp < 0):
                    pass
                elif (itp - self.mask_window // 2 >= 0) and (itp - self.mask_window // 2 < target.shape[0]):
                    target[itp - self.mask_window // 2:itp + self.mask_window // 2, 0, 1] = \
                        np.exp(-(np.arange(-self.mask_window // 2, self.mask_window // 2)) ** 2 / (
                                    2 * (self.mask_window // 4) ** 2))[:target.shape[0] - (itp - self.mask_window // 2)]
                    itp_true.append(itp)
                elif (itp - self.mask_window // 2 < target.shape[0]):
                    target[0:itp + self.mask_window // 2, 0, 1] = \
                        np.exp(-(np.arange(0, itp + self.mask_window // 2) - itp) ** 2 / (
                                    2 * (self.mask_window // 4) ** 2))[:target.shape[0] - (itp - self.mask_window // 2)]
                    itp_true.append(itp)

                if (its >= target.shape[0]) or (its < 0):
                    pass
                elif (its - self.mask_window // 2 >= 0) and (its - self.mask_window // 2 < target.shape[0]):
                    target[its - self.mask_window // 2:its + self.mask_window // 2, 0, 2] = \
                        np.exp(-(np.arange(-self.mask_window // 2, self.mask_window // 2)) ** 2 / (
                                    2 * (self.mask_window // 4) ** 2))[:target.shape[0] - (its - self.mask_window // 2)]
                    its_true.append(its)
                elif (its - self.mask_window // 2 < target.shape[0]):
                    target[0:its + self.mask_window // 2, 0, 2] = \
                        np.exp(-(np.arange(0, its + self.mask_window // 2) - its) ** 2 / (
                                    2 * (self.mask_window // 4) ** 2))[:target.shape[0] - (its - self.mask_window // 2)]
                    its_true.append(its)
            target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

            sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                              self.target_placeholder: target,
                                              self.fname_placeholder: fname,
                                              self.itp_placeholder: itp_true,
                                              self.its_placeholder: its_true})
        return 0


class DataReader_pred(DataReader):

    def __init__(self,
                 data_dir,
                 data_list,
                 queue_size,
                 coord,
                 input_length=None,
                 config=Config()):
        # TODO : use inheritence correctly
        self.config = config
        tmp_list = pd.read_csv(data_list, header=0)
        self.data_list = tmp_list
        self.num_data = len(self.data_list)
        self.data_dir = data_dir
        self.queue_size = queue_size
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        if input_length is not None:
            logging.warning("Using input length: {}".format(input_length))
            self.X_shape[0] = input_length
            self.Y_shape[0] = input_length

        self.coord = coord
        self.threads = []
        self.add_placeholder()

    def add_placeholder(self):
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                         ['float32', 'string'],
                                         shapes=[self.config.X_shape, []])

        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                           self.fname_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_up_to(num_elements)
        return output

    def thread_main(self, sess, n_threads=1, start=0):
        index = list(range(start, self.num_data, n_threads))
        for i in index:
            fname = self.data_list.iloc[i]['fname']
            fp = os.path.join(self.data_dir, fname)
            try:
                meta = np.load(fp)
            except:
                logging.error("Failed reading {}".format(fname))
                continue
            shift = 0
            # sample = meta['data'][shift:shift+self.X_shape, np.newaxis, :]
            sample = meta['data'][:, np.newaxis, :]
            if np.array(sample.shape).all() != np.array(self.X_shape).all():
                logging.error("{}: shape {} is not same as input shape {}!".format(fname, sample.shape, self.X_shape))
                continue

            if np.isnan(sample).any() or np.isinf(sample).any():
                logging.warning("Data error: {}\nReplacing nan and inf with zeros".format(fname))
                sample[np.isnan(sample)] = 0
                sample[np.isinf(sample)] = 0

            sample = self.normalize(sample)
            sample = self.adjust_missingchannels(sample)
            sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                              self.fname_placeholder: fname})


class DataReader_mseed(DataReader):

    def __init__(self, data_dir, data_list, queue_size, coord, input_length=3000, config=Config()):

        DataReader.__init__(
            self,
            data_dir=data_dir, data_list=data_list, mask_window=0,
            queue_size=queue_size, coord=coord, config=config)

        self.mask_window = None  # not used by this class
        self.input_length = config.X_shape[0]

        if input_length is not None:
            logging.warning("Using input length: {}".format(input_length))
            self.X_shape[0] = input_length
            self.Y_shape[0] = input_length
            self.input_length = input_length

    def add_placeholder(self):
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
        self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                         ['float32', 'string'],
                                         shapes=[self.config.X_shape, []])

        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                           self.fname_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_up_to(num_elements)
        return output

    def read_mseed(self, efile, nfile, zfile):
        """
        default mseed preprocessing here
        """
        estream = obspy.read(efile, format="MSEED")
        nstream = obspy.read(nfile, format="MSEED")
        zstream = obspy.read(zfile, format="MSEED")

        starttime, endtime = np.inf, -np.inf
        for st, expected_comp in zip([estream, nstream, zstream], 'ENZ'):
            for tr in st:
                if tr.stats.sampling_rate != 100.:
                    raise ValueError(
                        'Sampling rate was {}Hz'.format(tr.stats.samping_rate))
                if tr.stats.channel[2] != expected_comp:
                    raise ValueError(
                        'Channel was {} and I was expecting ??{}'.format(tr.stats.channel, expected_comp))

                starttime = np.min([starttime, tr.stats.starttime.timestamp])
                endtime = np.max([endtime, tr.stats.endtime.timestamp])

        for st in estream, nstream, zstream:
            st.detrend('constant')
            st.merge(fill_value=0)
            st.trim(UTCDateTime(starttime),
                    UTCDateTime(endtime), pad=True, fill_value=0.)
            assert len(st) == 1  # QC
            assert st[0].stats.samping_rate == estream[0].stats.samping_rate  # QC

        data = np.vstack([st[0].data for st in [estream, nstream, zstream]])

        start = zstream[0].stats.starttime
        nt = data.shape[1]
        dt = zstream[0].stats.delta
        timearray = start.timestamp + np.arange(nt) * dt

        ## can test small sampling rate for longer distance
        # meta = meta.interpolate(sampling_rate=100)

        pad_width = int((np.ceil((nt - 1) / self.input_length)) * self.input_length - nt)
        if pad_width == -1:
            data = data[:, :-1]
            nt -= 1
            timearray = timearray[:-1]
        else:
            # pad the data
            data = np.pad(data, ((0, 0), (0, pad_width)), 'constant', constant_values=(0, 0))
            # recompute the time array
            nt = data.shape[1]
            timearray = start.timestamp + np.arange(nt) * dt

        # repeat the data twice for 50% overlapping
        data = np.hstack([
            data,
            np.zeros_like(data[:, :self.input_length // 2]),
            data[:, :-self.input_length // 2]])

        # naive version, do exactly the same with time array as with the data
        # to ensure synchronization is preserved
        timearray = np.hstack([
            timearray,
            np.nan * np.zeros_like(data[:, :self.input_length // 2]),
            timearray[:-self.input_length // 2]])

        # one depth (axis 0) per component E, N, Z
        # then one raw per window
        # one column per sample in the window
        data = data.reshape((3, -1, self.input_length))
        timearray = timearray.reshape((3, -1, self.input_length))  # naive
        timearray = timearray[0, :, 0]  # keep only the starttime of each window in s since epoch

        # depths become the window numbers
        # lines become the samples inside the windows
        # columns become the component number
        # then a 1d axis is added in 2nd dimension
        data = data.transpose(1, 2, 0)[:, :, np.newaxis, :]

        return data, timearray

    def thread_main(self, sess, n_threads=1, start=0):
        # raise NotImplementedError('update this method so it can receive 3 indep mseed files for channels E, N, Z')
        for i in range(start, self.num_data, n_threads):
            #fname = self.data_list.iloc[i]['fname']
            #fp = os.path.join(self.data_dir, fname)

            network = self.data_list.iloc[i]['network']           # expects FR
            station = self.data_list.iloc[i]['station']           # expects ABCD
            location = self.data_list.iloc[i]['location']         # expects 00 or * or none
            dataquality = self.data_list.iloc[i]['dataquality']   # expects D or ?
            channel = self.data_list.iloc[i]['channel']           # expects EH* or EH? or HH? ...
            year = self.data_list.iloc[i]['year']                 # expects 2014
            julday = self.data_list.iloc[i]['julday']             # expects 014

            location = location.replace('none', '')
            assert len(location) == 2 or location == "*" or location == ""
            assert len(channel) == 3 and channel.endswith('?') or channel.endswith('*')

            filenames = []
            for comp in "ENZ":
                filepath = SDSPATH.format(
                    data_dir=self.data_dir, year=year, julday=julday,
                    dataquality=dataquality,
                    network=network, station=station,
                    location=location, channel=channel[:2] + comp)

                if os.path.isfile(filepath):
                    filenames.append(filepath)

                else:
                    import glob
                    ls = glob.iglob(filepath)
                    try:
                        filename = next(ls)
                    except StopIteration:
                        raise ValueError('no file responding to {}'.format(filepath))
                    finally:
                        ls.close()
                    filenames.append(filename)

            try:
                # meta = self.read_mseed(fp, [E, N, Z])
                data, timearray = self.read_mseed(
                    efile=filenames[0],
                    nfile=filenames[1],
                    zfile=filenames[2])

            except (IOError, ValueError, TypeError) as e:
                # you should never skip Exception, just notice the error type when it occurs and add it to the
                # list of ignored errors above
                logging.error("Failed reading mseed files {}".format(filenames))
                print(e)
                continue
            except BaseException as e:
                print('please never skip Exception or BaseException, '
                      'add the following type to the except close above : '
                      '{}'.format(e.__class__.__name__))
                raise e

            for i in tqdm(range(data.shape[0]), desc=r"{fp}"):
                # loop over windows
                sample = data[i]
                sample = self.normalize(sample)
                sample = self.adjust_missingchannels(sample)
                sess.run(self.enqueue,
                         feed_dict={self.sample_placeholder: sample,
                                    self.fname_placeholder: r"{fname}_{i * self.input_length}"})


if __name__ == "__main__":
    raise Exception
    ## debug
    data_reader = DataReader_mseed(
        data_dir="/data/beroza/zhuwq/Project-PhaseNet-mseed/mseed/",
        data_list="/data/beroza/zhuwq/Project-PhaseNet-mseed/fname.txt",
        queue_size=20,
        coord=None)
    data_reader.thread_main(None, n_threads=1, start=0)
# pred_fn(args, data_reader, log_dir=args.output_dir)
