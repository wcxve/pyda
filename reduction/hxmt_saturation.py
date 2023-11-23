import glob
import numpy as np
from astropy.io import fits
from astropy.time import Time


def utc_to_met(utc):
    utc = Time(utc, scale='utc', format='isot')
    utc0 = Time('2012-01-01T00:00:00', scale='utc', format='isot')
    return (utc - utc0).sec


def get_saturation(utc, tstart, tstop):
    path = '/hxmt/work/HXMT-DATA/1K'
    y = utc[:4]
    m = utc[5:7]
    d = utc[8:10]
    h = utc[11:13]
    files = glob.glob(f'{path}/Y{y}{m}/{y}{m}{d}-????/HXMT_{y}{m}{d}T{h}_HE-Evt_??????_V?_1K.FITS')
    file = sorted(files)[-1]
    t0 = utc_to_met(utc)
    with fits.open(file) as hdul:
        data = hdul['Events'].data
    time_mask = (t0 + tstart <= data['Time']) & (data['time'] <= t0 + tstop)
    t = data['Time'][time_mask] - t0
    det = data['Det_ID'][time_mask]
    t1 = t[(0 <= det) & (det <= 5)]
    t2 = t[(6 <= det) & (det <= 11)]
    t3 = t[(12 <= det) & (det <= 17)]
    tbins = np.arange(tstart, tstop, 0.005)
    counts1 = np.histogram(t1, tbins)[0]
    counts2 = np.histogram(t2, tbins)[0]
    counts3 = np.histogram(t3, tbins)[0]
    saturate1 = counts1 == 0
    saturate2 = counts2 == 0
    saturate3 = counts3 == 0
    saturate = saturate1 | saturate2 | saturate3
    if saturate.any():
        n = len(saturate1)
        nsat1 = saturate1.sum()
        nsat2 = saturate2.sum()
        nsat3 = saturate3.sum()
        nsat = saturate.sum()
        return 1.0 - nsat1/n, 1.0 - nsat2/n, 1.0 - nsat3/n, 1.0 - nsat/n
    else:
        return 1.0, 1.0, 1.0, 1.0


if __name__ == '__main__':
    utc_list =[
        '2022-10-15T15:40:28.600',
        '2022-10-16T10:33:30.960',
        '2022-10-16T10:43:17.070',
        '2022-10-16T12:03:05.975',
        '2022-10-16T12:17:36.540',
        '2022-10-16T13:08:00.900',
        '2022-10-16T13:45:47.265',
        '2022-10-16T15:02:57.175',
        '2022-10-17T05:42:37.165',
        '2022-10-17T07:38:50.670',
        '2022-10-17T09:46:05.995',
        '2022-10-17T11:14:42.175',
        '2022-10-17T11:19:24.160',
        '2022-10-17T11:21:27.715',
        '2022-10-17T11:30:16.120',
        '2022-10-17T11:39:01.160',
        '2022-10-17T11:52:51.480',
        '2022-10-17T12:58:41.760',
        '2022-10-17T13:00:10.940',
        '2022-10-17T13:15:44.205',
        '2022-10-17T13:51:54.620',
        '2022-10-17T13:56:25.445',
        '2022-10-17T14:00:08.375',
        '2022-10-17T14:30:36.490',
        '2022-10-17T14:36:10.635',
        '2022-10-17T14:39:32.960',
        '2022-10-17T15:04:25.035',
        '2022-10-17T15:11:41.190',
        '2022-10-17T15:14:07.380',
        '2022-10-17T15:22:56.580',
        '2022-10-17T15:32:20.480',
        '2022-10-17T15:34:35.875',
        '2022-10-17T16:17:17.125',
        '2022-10-17T16:18:19.085',
        '2022-10-18T15:03:07.950',
        '2022-10-19T06:24:50.190',
        '2022-10-19T07:08:19.760',
        '2022-10-19T07:55:19.600',
        '2022-10-19T08:35:46.215',
        '2022-10-19T08:41:40.140',
        '2022-10-19T09:32:26.680',
        '2022-10-19T09:47:18.225',
        '2022-10-19T13:19:54.075',
        '2022-10-20T09:26:41.200',
        '2022-10-21T07:49:21.150',
        '2022-10-21T10:01:45.000',
        '2022-10-22T10:00:06.815',
        '2022-10-24T13:42:48.175'
    ]
    saturate = np.empty((len(utc_list), 4))
    for i in range(len(utc_list)):
        si = get_saturation(utc_list[i], -2, 2)
        saturate[i] = si
        print(i+1, si)
    saturate = np.round(saturate, 3)
    np.savetxt('HE_saturation.txt', np.column_stack([utc_list, saturate]), fmt='%s', delimiter='\t')
