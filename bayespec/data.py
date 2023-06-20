import os
import warnings

import numpy as np
import xarray as xr
from astropy.io import fits

__all__ = ['Data']


class Data:
    # TODO: extract rsp2 file
    # Currently need to manually specified in respfile and ancrfile.
    def __init__(
        self, erange, specfile,
        backfile=None, respfile=None, ancrfile=None, name=None,
        statistic='chi', ignore_bad=True, keep_channel_info=False
    ):
        if statistic not in ['chi', 'cstat', 'pstat', 'pgstat', 'wstat']:
            raise ValueError(
                'available likelihood statistics are chi, cstat, pstat, pgstat'
                ' and wstat'
            )
        else:
            self.statistic = statistic

        self._extract_spec(specfile)
        self._set_name(name)
        self._extract_back(backfile)
        self._extract_resp(respfile, ancrfile)
        self._filter_channel(erange, ignore_bad, keep_channel_info)

    def _extract_spec(self, specfile):
        if '{' in specfile and specfile[-1] == '}':
            self._spec_num = int(specfile.split('{')[1].split('}')[0]) - 1
            self._spec_typeii = True
            specfile = specfile.split('{')[0]
        else:
            self._spec_typeii = False

        if not os.path.exists(specfile):
            raise FileNotFoundError(f'spectrum file "{specfile}" not found')

        with fits.open(specfile) as spec_hdul:
            spec_header = spec_hdul['SPECTRUM'].header
            spec_data = spec_hdul['SPECTRUM'].data

        spec_poisson = spec_header['POISSERR']

        if self._spec_typeii:
            spec_data = spec_data[self._spec_num : self._spec_num + 1]
            spec_exposure = spec_data['EXPOSURE'][0]

            if 'COUNTS' in spec_data.names:
                spec_counts = spec_data['COUNTS'][0]
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR'][0]
            elif 'RATE' in spec_data.names:
                spec_counts = spec_data['RATE'][0] * spec_exposure
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR'][0] * spec_exposure
            else:
                raise ValueError(
                    f'Spectrum ({specfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in spec_data.names:
                spec_quality = spec_data['QUALITY'][0]
            else:
                spec_quality = np.zeros(len(spec_counts))

            if 'GROUPING' in spec_data.names:
                grouping = np.flatnonzero(spec_data['GROUPING'][0] == 1)
            else:
                grouping = np.arange(len(spec_counts))

        else:
            spec_exposure = spec_header['EXPOSURE']

            if 'COUNTS' in spec_data.names:
                spec_counts = spec_data['COUNTS']
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR']
            elif 'RATE' in spec_data.names:
                spec_counts = spec_data['RATE'] * spec_exposure
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR'] * spec_exposure
            else:
                raise ValueError(
                    f'Spectrum ({specfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in spec_data.names:
                spec_quality = spec_data['QUALITY']
            else:
                spec_quality = np.zeros(len(spec_counts))

            if 'GROUPING' in spec_data.names:
                grouping = np.flatnonzero(spec_data['GROUPING'] == 1)
            else:
                grouping = np.arange(len(spec_counts))


        if spec_poisson:  # check if counts are integers
            diff = np.abs(spec_counts - np.round(spec_counts))
            if np.any(diff > 1e-8 * spec_counts):
                warnings.warn(
                    f'spectrum ({specfile}) counts are not integers, which'
                    ' could lead to wrong result',
                    stacklevel=2
                )

        self.spec_exposure = spec_exposure
        self.spec_poisson = spec_poisson
        self._spec_header = spec_header
        self._spec_data = spec_data
        self._spec_counts = np.array(spec_counts, dtype=np.float64)
        self._spec_error = np.array(spec_error, dtype=np.float64)
        self._spec_quality = np.array(spec_quality, dtype=np.int8)
        self._grouping = np.array(grouping, dtype=np.int64)


    def _extract_back(self, backfile):
        if self._spec_typeii:
            backfile = backfile or self._spec_data['BACKFILE'][0]
        else:
            backfile = backfile or self._spec_header['BACKFILE']

        if backfile.lower() in ['none', '']:
            warnings.warn(
                f'assumes {self.name} has no background',
                stacklevel=2
            )
            self.has_back = False
            return None

        if '{' in backfile and backfile[-1] == '}':
            self._back_num = int(backfile.split('{')[1].split('}')[0]) - 1
            self._back_typeii = True
            backfile = backfile.split('{')[0]
        else:
            self._back_typeii = False

        if not os.path.exists(backfile):
            raise FileNotFoundError(f'background file "{backfile}" not found')

        with fits.open(backfile) as back_hdul:
            back_header = back_hdul['SPECTRUM'].header
            back_data = back_hdul['SPECTRUM'].data

        back_poisson = back_header['POISSERR']

        if self._back_typeii:
            back_data = back_data[self._back_num : self._back_num + 1]

            back_exposure = back_data['EXPOSURE'][0]

            if 'COUNTS' in back_data.names:
                back_counts = back_data['COUNTS'][0]
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR'][0]
            elif 'RATE' in back_data.names:
                back_counts = back_data['RATE'][0] * back_exposure
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR'][0] * back_exposure
            else:
                raise ValueError(
                    f'Background ({backfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in back_data.names:
                back_quality = back_data['QUALITY'][0]
            else:
                back_quality = np.zeros(len(back_counts))

        else:
            back_exposure = back_header['EXPOSURE']

            if 'COUNTS' in back_data.names:
                back_counts = back_data['COUNTS']
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR']
            elif 'RATE' in back_data.names:
                back_counts = back_data['RATE'] * back_exposure
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR'] * back_exposure
            else:
                raise ValueError(
                    f'Background ({backfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in back_data.names:
                back_quality = back_data['QUALITY']
            else:
                back_quality = np.zeros(len(back_counts))


        if back_poisson:  # check if counts are integers
            diff = np.abs(back_counts - np.round(back_counts))
            if np.any(diff > 1e-8 * back_counts):
                warnings.warn(
                    f'background ({backfile}) counts are not integers, '
                    'which could lead to wrong result',
                    stacklevel=2
                )

        self.has_back = True
        self.back_exposure = back_exposure
        self.back_poisson = back_poisson
        self._back_header = back_header
        self._back_data = back_data
        self._back_counts = np.array(back_counts, dtype=np.float64)
        self._back_error = np.array(back_error, dtype=np.float64)
        self._back_quality = np.array(back_quality, dtype=np.int8)


    def _extract_resp(self, respfile, ancrfile):
        if self._spec_typeii:
            respfile = respfile or self._spec_data['RESPFILE'][0]
            if ancrfile:
                ancrfile = ancrfile
            else:
                if 'ANCRFILE' in self._spec_data.names:
                    ancrfile = self._spec_data['ANCRFILE'][0]
                else:
                    ancrfile = self._spec_header['ANCRFILE']
        else:
            respfile = respfile or self._spec_header['RESPFILE']
            ancrfile = ancrfile or self._spec_header['ANCRFILE']

        if respfile.lower() in ['none', '']:
            raise FileNotFoundError('response file is required')
        elif not os.path.exists(respfile):
            raise FileNotFoundError(f'response file "{respfile}" not found')
        else:
            with fits.open(respfile) as resp_hdul:
                ebounds = resp_hdul['EBOUNDS'].data
                if 'MATRIX' in [hdu.name for hdu in resp_hdul]:
                    resp = resp_hdul['MATRIX'].data
                else:
                    resp = resp_hdul['SPECRESP MATRIX'].data

                if len(ebounds) != len(self._spec_counts):
                    raise ValueError('response is not match with spectrum')

                self._channel = ebounds['CHANNEL']

        # a simple wrap around for zero elements for some response files
        mask = [np.any(i['MATRIX'] > 0.0) for i in resp]
        resp_ = resp[mask]

        # assumes ph_ebins is continuous
        ph_ebins = np.append(resp_['ENERG_LO'], resp_['ENERG_HI'][-1])
        ch_ebins = np.column_stack((ebounds['E_MIN'], ebounds['E_MAX']))

        # extract response matrix
        resp_matrix = resp_['MATRIX']
        if resp_matrix.dtype == np.dtype('O'):
            resp_matrix = np.array(resp_matrix.tolist(), dtype=np.float64)

        if ancrfile.lower() in ['none', '']:
            pass
        elif not os.path.exists(ancrfile):
            raise FileNotFoundError(f'arf file "{ancrfile}" not found')
        else:
            with fits.open(ancrfile) as arf_hdul:
                arf_data = arf_hdul['SPECRESP'].data['SPECRESP']

            if len(arf_data) != len(resp.data):
                raise ValueError(
                    f'arf ({ancrfile}) is not matched with rmf ({respfile})'
                )

            resp_matrix = np.expand_dims(arf_data[mask], axis=1) * resp_matrix

        ichannel = [f'{self.name}_In{c}' for c in range(len(resp_matrix))]
        self.ichannel = ichannel
        self.ph_ebins = ph_ebins
        self._ch_ebins = ch_ebins
        self._resp_matrix = resp_matrix


    def _set_name(self, name):
        excluded = ['', 'none', 'unknown']
        if name:
            self.name = name
        elif (_:=self._spec_header['DETNAM']) and _.lower() not in excluded:
            self.name = self._spec_header['DETNAM']
        elif (_:=self._spec_header['INSTRUME']) and _.lower() not in excluded:
            self.name = self._spec_header['INSTRUME']
        elif (_:=self._spec_header['TELESCOP']) and _.lower() not in excluded:
            self.name = self._spec_header['TELESCOP']
        else:
            raise ValueError('input for `name` is required')


    def _filter_channel(self, erange, ignore_bad, keep_channel_info):
        if ignore_bad:
            bad_quality = [1, 5]
        else:
            bad_quality = [1]

        good_quality = ~np.isin(self._spec_quality, bad_quality)
        if self.has_back:
            good_quality &= ~np.isin(self._back_quality, bad_quality)

        any_good_in_group = np.add.reduceat(good_quality, self._grouping) != 0

        factor = np.where(good_quality, 1.0, 0.0)
        resp_matrix = np.add.reduceat(
            self._resp_matrix * factor,
            self._grouping,
            axis=1
        )
        resp_matrix = resp_matrix[:, any_good_in_group]

        groups_edge_indices = np.append(self._grouping, len(self._spec_counts))
        channel = self._channel
        emin, emax = self._ch_ebins.T
        groups_channel = []
        groups_emin = []
        groups_emax = []
        for i in range(len(self._grouping)):
            if not any_good_in_group[i]:
                continue
            slice_i = slice(groups_edge_indices[i], groups_edge_indices[i + 1])
            quality_slice = good_quality[slice_i]
            channel_slice = channel[slice_i]
            groups_channel.append(channel_slice[quality_slice].astype(str))
            groups_emin.append(min(emin[slice_i]))
            groups_emax.append(max(emax[slice_i]))

        if keep_channel_info:
            groups_channel = np.array([
                self.name+'_Ch' + '+'.join(c)
                for c in groups_channel
            ])
        else:
            groups_channel = np.array([
                self.name+f'_Ch{c}'
                for c in np.flatnonzero(any_good_in_group)
            ])

        groups_ch_ebins = np.column_stack((groups_emin, groups_emax))
        groups_emin = np.array(groups_emin)
        groups_emax = np.array(groups_emax)

        erange = np.atleast_2d(erange)
        emin = np.expand_dims(erange[:, 0], axis=1)
        emax = np.expand_dims(erange[:, 1], axis=1)
        chmask = (emin <= groups_emin) & (groups_emax <= emax)
        chmask = np.any(chmask, axis=0)

        self.channel = groups_channel[chmask]
        self.ch_ebins = groups_ch_ebins[chmask]
        self.resp_matrix = resp_matrix[:, chmask]

        spec_counts = np.where(good_quality, self._spec_counts, 0)
        spec_counts = np.add.reduceat(spec_counts, self._grouping)
        spec_counts = spec_counts[any_good_in_group]

        spec_error = np.where(good_quality, self._spec_error, 0)
        spec_error = np.sqrt(
            np.add.reduceat(spec_error*spec_error, self._grouping)
        )
        spec_error = spec_error[any_good_in_group]

        self.spec_counts = spec_counts[chmask]
        self.spec_error = spec_error[chmask]

        self.data = xr.Dataset(
            data_vars={
                'name': self.name,
                'spec_counts': ('channel', self.spec_counts),
                'spec_error': ('channel', self.spec_error),
                'spec_poisson': self.spec_poisson,
                'spec_exposure': self.spec_exposure,
                'ph_ebins': self.ph_ebins,
                'ch_ebins': (['channel', 'edge'], self.ch_ebins),
                'resp_matrix': (['channel_in', 'channel'], self.resp_matrix),
            },
            coords={
                'channel_in': self.ichannel,
                'channel': self.channel,
                'edge': ['start', 'stop']
            }
        )

        if self.has_back:
            back_counts = np.where(good_quality, self._back_counts, 0)
            back_counts = np.add.reduceat(back_counts, self._grouping)
            back_counts = back_counts[any_good_in_group]

            back_error = np.where(good_quality, self._back_error, 0)
            back_error = np.sqrt(
                np.add.reduceat(back_error * back_error, self._grouping)
            )
            back_error = back_error[any_good_in_group]

            self.back_counts = back_counts[chmask]
            self.back_error = back_error[chmask]

            self.data['back_counts'] = ('channel', self.back_counts)
            self.data['back_error'] = ('channel', self.back_error)
            self.data['back_poisson'] = self.back_poisson
            self.data['back_exposure'] = self.back_exposure


def to_bayespec_data(
    n_on, err_on, t_on, n_off, err_off, t_off, channel, respfile,
    is_on_poisson=True, is_off_poisson=True, name=None
):
    if not os.path.exists(respfile):
        raise FileNotFoundError(f'response file "{respfile}" not found')

    with fits.open(respfile) as hdul:
        if 'MATRIX' in [hdu.name for hdu in hdul]:
            matrix = hdul['MATRIX'].data
            name = name or hdul['MATRIX'].header['INSTRUME']
        else:
            matrix = hdul['SPECRESP MATRIX'].data
            name = name or hdul['SPECRESP MATRIX'].header['INSTRUME']
        ebounds = hdul['EBOUNDS'].data

    mask = [np.any(i['MATRIX'] > 0.0) for i in matrix]
    matrix = matrix[mask]

    ph_ebins = np.append(matrix['ENERG_LO'], matrix['ENERG_HI'][-1])

    rsp = matrix['MATRIX']
    if rsp.dtype is np.dtype('O'):
        rsp = np.asarray(rsp.tolist())


    ch_ebins = np.append(ebounds['E_MIN'], ebounds['E_MAX'][-1])
    ch_ebins = np.column_stack((ch_ebins[:-1], ch_ebins[1:]))

    chmask = np.isin(ebounds['CHANNEL'], channel)

    n_on = np.array(n_on, dtype=np.float64)
    err_on = np.array(err_on, dtype=np.float64)
    if n_off is not None:
        n_off = np.array(n_off, dtype=np.float64)
        err_off = np.array(err_off, dtype=np.float64)
    ph_ebins = np.array(ph_ebins, dtype=np.float64)
    ch_ebins = np.array(ch_ebins[chmask], dtype=np.float64)
    rsp = np.array(rsp[:, chmask], dtype=np.float64)

    data_set = xr.Dataset(
        data_vars={
            'name': name,
            'n_on': ('channel', n_on),
            'n_off': ('channel', n_off) if n_off is not None else n_off,
            'err_on': ('channel', err_on),
            'err_off': ('channel', err_off) if n_off is not None else err_off,
            'is_on_poisson': is_on_poisson,
            'is_off_poisson': is_off_poisson,
            't_on': t_on,
            't_off': t_off,
            'ph_ebins': ph_ebins,
            'ch_ebins': (['channel', 'edge'], ch_ebins),
            'response': (['channel_in', 'channel'], rsp),
        },
        coords={
            'channel_in': [f'{name}_I{c}' for c in range(len(rsp))],
            'channel': [f'{name}_D{c}' for c in channel],
            'edge': ['start', 'stop']
        }
    )

    return data_set