import numpy as np
from astropy.io import fits

import pyda

__all__ = ['gecam_phaii_poisson']


def gecam_phaii_gaussian(
    outpath, tehist, backfile=None, respfile=None, spectype='TOTAL'
):
    tstart, tstop = tehist.tbins.values.T
    channel = tehist.channel.values
    emin, emax = tehist.ebins.values.T
    counts = tehist.counts.values.T
    error = tehist.counts_error.values.T
    exposure = tehist.exposure.values

    nchan = 498 if tehist.telescope != 'GECAM-C' else 470
    channel_ = np.arange(nchan)
    counts_ = np.full((tstart.size, nchan), np.nan, dtype=np.float64)
    counts_[:, channel] = counts
    error_ = np.full((tstart.size, nchan), np.nan, dtype=np.float64)
    error_[:, channel] = error
    quality = np.full_like(counts_, 1, dtype=np.int64)
    quality[:, channel] = 0
    emin_ = np.full(nchan, np.nan)
    emin_[channel] = emin
    emax_ = np.full(nchan, np.nan)
    emax_[channel] = emax
    telescope = str(tehist.telescope.values)
    instrument = str(tehist.instrument.values)
    detname = str(tehist.detname.values)


    generate_phaii_gaussian_error(
        outpath, tstart, tstop, channel_, emin_, emax_, counts_, error_,
        exposure, telescope, instrument, detname,
        quality=quality, group=None, backfile=backfile, respfile=respfile,
        chantype='PI', spectype=spectype
    )


def gecam_phaii_poisson(
    outpath, tehist, backfile=None, respfile=None, spectype='TOTAL'
):
    tstart, tstop = tehist.tbins.values.T
    channel = tehist.channel.values
    emin, emax = tehist.ebins.values.T
    counts = tehist.counts.values.T
    exposure = tehist.exposure.values

    nchan = 498 if tehist.telescope != 'GECAM-C' else 470
    channel_ = np.arange(nchan)
    counts_ = np.full((tstart.size, nchan), 0, dtype=np.int64)
    counts_[:, channel] = counts
    quality = np.full_like(counts_, 1, dtype=np.int64)
    quality[:, channel] = 0
    emin_ = np.full(nchan, np.nan)
    emin_[channel] = emin
    emax_ = np.full(nchan, np.nan)
    emax_[channel] = emax
    telescope = str(tehist.telescope.values)
    instrument = str(tehist.instrument.values)
    detname = str(tehist.detname.values)


    generate_phaii_poisson_error(
        outpath, tstart, tstop, channel_, emin_, emax_, counts_, exposure,
        telescope, instrument, detname,
        quality=quality, group=None, backfile=backfile, respfile=respfile,
        chantype='PI', spectype=spectype
    )


def generate_phaii_gaussian_error(
    specfile, tstart, tstop, channel, emin, emax, counts, error, exposure,
    telescope, instrument, detname,
    quality=None, group=None, backfile=None, respfile=None, chantype='PI',
    spectype='TOTAL'
):
    r"""
    Generate spectral file containing counts data.

    Parameters
    ----------
    specfile : str
    tstart : (t,) array_like
    tstop : (t,) array_like
    channel : (c,) array_like
    emin : (c,) array_like
    emax : (c,) array_like
    counts : (t, c) array_like
    error : (t, c) array_like
    exposure : (t,) array_like
    telescope : str
    instrument : str
    detname : str
    quality : None or (t, c) array_like
    group : None or (t, c) array_like
    backfile : None or (t,) array_like
    respfile : None or (t,) array_like

    """
    tstart = np.atleast_1d(np.array(tstart, dtype=np.float64))
    tstop = np.atleast_1d(np.array(tstop, dtype=np.float64))
    channel = np.array(channel, dtype=np.int64)
    emin = np.array(emin, dtype=np.float64)
    emax = np.array(emax, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)
    error = np.array(error, dtype=np.float64)
    exposure = np.atleast_1d(np.array(exposure, dtype=np.float64))

    if not tstart.shape == tstop.shape == exposure.shape:
        raise ValueError(
            f'tstart {tstart.shape}, tstop {tstop.shape}, and exposure '
            f'{exposure.shape} are not matched'
        )
    if not channel.shape == emin.shape == emax.shape == (counts.shape[1],):
        raise ValueError(
            f'channel {channel.shape}, emin {emin.shape}, emax {emax.shape} '
            f'and counts {counts.shape} are not matched'
        )

    if not counts.shape == error.shape:
        raise ValueError(
            f'counts {counts.shape} and error {error.shape} are not matched'
        )

    if not counts.shape[0] == tstart.shape[0]:
        raise ValueError(
            f'counts {counts.shape} and time {tstart.shape} are not matched'
        )

    if group is not None:
        group = np.array(group, dtype=np.int64)
        if not group.shape == counts.shape:
            raise ValueError(
                f'group {group.shape} and counts {counts.shape} are not '
                'matched'
            )
    else:
        group = np.ones_like(counts)

    if quality is not None:
        quality = np.array(quality, dtype=np.int64)
        if not quality.shape == counts.shape:
            raise ValueError(
                f'quality {quality.shape} and counts {counts.shape} are not '
                'matched'
            )
    else:
        quality = np.zeros_like(counts)

    if backfile is not None:
        backfile = np.array(backfile, dtype=str)
        if not backfile.shape == tstart.shape:
            raise ValueError(
                f'backfile {backfile.shape} and tstart {tstart.shape} are not '
                'matched'
            )
    else:
        backfile = np.array(['' for _ in range(tstart.size)], dtype=str)

    if respfile is not None:
        if type(respfile) is str:
            respfile = np.array([respfile for _ in range(tstart.size)],
                                dtype=str)
        else:
            respfile = np.array(respfile, dtype=str)
            if not respfile.shape == tstart.shape:
                raise ValueError(
                    f'respfile {respfile.shape} and tstart {tstart.shape} are '
                    'not matched'
                )
    else:
        respfile = np.array(['' for _ in range(tstart.size)], dtype=str)


    primary = fits.PrimaryHDU()
    creator = f"{pyda.__name__}_v{pyda.__version__}"
    primary.header['CREATOR'] = (creator, 'Software and version creating file')
    primary.header['FILETYPE'] = ('PHAII', 'Name for this type of FITS file')
    primary.header['FILE-VER'] = ('1.0.0', 'Version of the format for this filetype')
    primary.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    primary.header['INSTRUME'] = (instrument, 'Specific instrument used for observation')
    primary.header['DETNAM'] = (detname, 'Individual detector name')
    primary.header['FILENAME'] = (specfile.split('/')[-1], 'Name of this file')

    ebounds_columns = [
        fits.Column(name='CHANNEL', format='1I', array=channel),
        fits.Column(name='E_MIN', format='1E', unit='keV', array=emin),
        fits.Column(name='E_MAX', format='1E', unit='keV', array=emax)
    ]
    ebounds = fits.BinTableHDU.from_columns(ebounds_columns)
    ebounds.header['EXTNAME'] = ('EBOUNDS', 'Name of this binary table extension')
    ebounds.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    ebounds.header['INSTRUME'] = (instrument, 'Specific instrument used for observation')
    ebounds.header['DETNAM'] = (detname, 'Individual detector name')
    ebounds.header['HDUCLASS'] = ('OGIP', 'Conforms to OGIP standard indicated in HDUCLAS1')
    ebounds.header['HDUCLAS1'] = ('RESPONSE', 'These are typically found in RMF files')
    ebounds.header['HDUCLAS2'] = ('EBOUNDS', 'From CAL/GEN/92-002')
    ebounds.header['HDUVERS'] = ('1.2.1', 'Version of HDUCLAS1 format in use')
    ebounds.header['CHANTYPE'] = (chantype, 'Channel type')
    ebounds.header['DETCHANS'] = (len(channel), 'Total number of channels in each rate')

    spectrum_columns = [
        fits.Column(name='TIME', format='1D', unit='s', array=tstart),
        fits.Column(name='ENDTIME', format='1D', unit='s', array=tstop),
        fits.Column(name='EXPOSURE', format='1E', unit='s', array=exposure),
        fits.Column(name='COUNTS', format=f'{len(channel)}D', array=counts),
        fits.Column(name='STAT_ERR', format=f'{len(channel)}D', array=error),
        fits.Column(name='QUALITY', format=f'{len(channel)}I', array=quality),
        fits.Column(name='GROUPING', format=f'{len(channel)}I', array=group),
        fits.Column(name='BACKFILE', format='150A', array=backfile),
        fits.Column(name='RESPFILE', format='150A', array=respfile)
    ]
    spectrum = fits.BinTableHDU.from_columns(spectrum_columns)
    spectrum.header['EXTNAME'] = ('SPECTRUM', 'Name of this binary table extension')
    spectrum.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    spectrum.header['INSTRUME'] = (instrument, 'Specific instrument used for observation')
    spectrum.header['DETNAM'] = (detname, 'Individual detector name')
    spectrum.header['AREASCAL'] = (1.0, 'No special scaling of effective area by channel')
    spectrum.header['BACKSCAL'] = (1.0, 'No scaling of background')
    spectrum.header['CORRSCAL'] = (1.0, 'Correction scaling file')
    spectrum.header['ANCRFILE'] = ('none', 'Name of corresponding ARF file (if any)')
    spectrum.header['SYS_ERR'] = (0.0, 'No systematic errors')
    spectrum.header['POISSERR'] = (False, 'Assume Poisson Errors')
    spectrum.header['STATERR'] = (True, 'Statistical errors specified')
    spectrum.header['GROUPING'] = (1, 'Grouping of the data has been defined')
    spectrum.header['QUALITY'] = (1, 'Data quality information specified')
    spectrum.header['HDUCLASS'] = ('OGIP', 'Conforms to OGIP standard indicated in HDUCLAS1')
    spectrum.header['HDUCLAS1'] = ('SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)')
    spectrum.header['HDUCLAS2'] = (spectype, 'Indicates TOTAL/NET/BKG data')
    spectrum.header['HDUCLAS3'] = ('COUNT', 'Indicates data stored as counts')
    spectrum.header['HDUCLAS4'] = ('TYPEII', 'Indicates PHA Type II file format')
    spectrum.header['HDUVERS'] = ('1.2.1', 'Version of HDUCLAS1 format in use')
    spectrum.header['CHANTYPE'] = (chantype, 'Channel type')
    spectrum.header['DETCHANS'] = (len(channel), 'Total number of channels in each rate')

    fits.HDUList([primary, ebounds, spectrum]).writeto(specfile, overwrite=True)


def generate_phaii_poisson_error(
    specfile, tstart, tstop, channel, emin, emax, counts, exposure,
    telescope, instrument, detname,
    quality=None, group=None, backfile=None, respfile=None, chantype='PI',
    spectype='TOTAL'
):
    r"""
    Generate spectral file containing counts data.

    Parameters
    ----------
    specfile : str
    tstart : (t,) array_like
    tstop : (t,) array_like
    channel : (c,) array_like
    emin : (c,) array_like
    emax : (c,) array_like
    counts : (t, c) array_like
    exposure : (t,) array_like
    telescope : str
    instrument : str
    detname : str
    quality : None or (t, c) array_like
    group : None or (t, c) array_like
    backfile : None or (t,) array_like
    respfile : None or (t,) array_like

    """
    tstart = np.atleast_1d(np.array(tstart, dtype=np.float64))
    tstop = np.atleast_1d(np.array(tstop, dtype=np.float64))
    channel = np.array(channel, dtype=np.int64)
    emin = np.array(emin, dtype=np.float64)
    emax = np.array(emax, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)
    exposure = np.atleast_1d(np.array(exposure, dtype=np.float64))

    if not tstart.shape == tstop.shape == exposure.shape:
        raise ValueError(
            f'tstart {tstart.shape}, tstop {tstop.shape}, and exposure '
            f'{exposure.shape} are not matched'
        )
    if not channel.shape == emin.shape == emax.shape == (counts.shape[1],):
        raise ValueError(
            f'channel {channel.shape}, emin {emin.shape}, emax {emax.shape} '
            f'and counts {counts.shape} are not matched'
        )

    if not counts.shape[0] == tstart.shape[0]:
        raise ValueError(f'counts {counts.shape} and time {tstart.shape[0]})')

    if group is not None:
        group = np.array(group, dtype=np.int64)
        if not group.shape == counts.shape:
            raise ValueError(
                f'group {group.shape} and counts {counts.shape} are not '
                'matched'
            )
    else:
        group = np.ones_like(counts)

    if quality is not None:
        quality = np.array(quality, dtype=np.int64)
        if not quality.shape == counts.shape:
            raise ValueError(
                f'quality {quality.shape} and counts {counts.shape} are not '
                'matched'
            )
    else:
        quality = np.zeros_like(counts)

    if backfile is not None:
        if type(backfile) is str:
            backfile = np.array(
                [backfile+f'{{{i+1}}}' for i in range(tstart.size)],
                dtype=str
        )
        else:
            backfile = np.array(backfile, dtype=str)
            if not backfile.shape == tstart.shape:
                raise ValueError(
                    f'backfile {backfile.shape} and tstart {tstart.shape} are '
                    'not matched'
                )
    else:
        backfile = np.array(['' for _ in range(tstart.size)], dtype=str)

    if respfile is not None:
        if type(respfile) is str:
            respfile = np.array([respfile for _ in range(tstart.size)],
                                dtype=str)
        else:
            respfile = np.array(respfile, dtype=str)
            if not respfile.shape == tstart.shape:
                raise ValueError(
                    f'respfile {respfile.shape} and tstart {tstart.shape} are '
                    'not matched'
                )
    else:
        respfile = np.array(['' for _ in range(tstart.size)], dtype=str)


    primary = fits.PrimaryHDU()
    creator = f"{pyda.__name__}_v{pyda.__version__}"
    primary.header['CREATOR'] = (creator, 'Software and version creating file')
    primary.header['FILETYPE'] = ('PHAII', 'Name for this type of FITS file')
    primary.header['FILE-VER'] = ('1.0.0', 'Version of the format for this filetype')
    primary.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    primary.header['INSTRUME'] = (instrument, 'Specific instrument used for observation')
    primary.header['DETNAM'] = (detname, 'Individual detector name')
    primary.header['FILENAME'] = (specfile.split('/')[-1], 'Name of this file')

    ebounds_columns = [
        fits.Column(name='CHANNEL', format='1I', array=channel),
        fits.Column(name='E_MIN', format='1E', unit='keV', array=emin),
        fits.Column(name='E_MAX', format='1E', unit='keV', array=emax)
    ]
    ebounds = fits.BinTableHDU.from_columns(ebounds_columns)
    ebounds.header['EXTNAME'] = ('EBOUNDS', 'Name of this binary table extension')
    ebounds.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    ebounds.header['INSTRUME'] = (instrument, 'Specific instrument used for observation')
    ebounds.header['DETNAM'] = (detname, 'Individual detector name')
    ebounds.header['HDUCLASS'] = ('OGIP', 'Conforms to OGIP standard indicated in HDUCLAS1')
    ebounds.header['HDUCLAS1'] = ('RESPONSE', 'These are typically found in RMF files')
    ebounds.header['HDUCLAS2'] = ('EBOUNDS', 'From CAL/GEN/92-002')
    ebounds.header['HDUVERS'] = ('1.2.1', 'Version of HDUCLAS1 format in use')
    ebounds.header['CHANTYPE'] = (chantype, 'Channel type')
    ebounds.header['DETCHANS'] = (len(channel), 'Total number of channels in each rate')

    spectrum_columns = [
        fits.Column(name='TIME', format='1D', unit='s', array=tstart),
        fits.Column(name='ENDTIME', format='1D', unit='s', array=tstop),
        fits.Column(name='EXPOSURE', format='1E', unit='s', array=exposure),
        fits.Column(name='COUNTS', format=f'{len(channel)}D', array=counts),
        fits.Column(name='QUALITY', format=f'{len(channel)}I', array=quality),
        fits.Column(name='GROUPING', format=f'{len(channel)}I', array=group),
        fits.Column(name='BACKFILE', format='150A', array=backfile),
        fits.Column(name='RESPFILE', format='150A', array=respfile)
    ]
    spectrum = fits.BinTableHDU.from_columns(spectrum_columns)
    spectrum.header['EXTNAME'] = ('SPECTRUM', 'Name of this binary table extension')
    spectrum.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    spectrum.header['INSTRUME'] = (instrument, 'Specific instrument used for observation')
    spectrum.header['DETNAM'] = (detname, 'Individual detector name')
    spectrum.header['AREASCAL'] = (1.0, 'No special scaling of effective area by channel')
    spectrum.header['BACKSCAL'] = (1.0, 'No scaling of background')
    spectrum.header['CORRSCAL'] = (1.0, 'Correction scaling file')
    spectrum.header['ANCRFILE'] = ('none', 'Name of corresponding ARF file (if any)')
    spectrum.header['SYS_ERR'] = (0.0, 'No systematic errors')
    spectrum.header['POISSERR'] = (True, 'Assume Poisson Errors')
    spectrum.header['GROUPING'] = (1, 'Grouping of the data has been defined')
    spectrum.header['QUALITY'] = (1, 'Data quality information specified')
    spectrum.header['HDUCLASS'] = ('OGIP', 'Conforms to OGIP standard indicated in HDUCLAS1')
    spectrum.header['HDUCLAS1'] = ('SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)')
    spectrum.header['HDUCLAS2'] = (spectype, 'Indicates TOTAL/NET/BKG data')
    spectrum.header['HDUCLAS3'] = ('COUNT', 'Indicates data stored as counts')
    spectrum.header['HDUCLAS4'] = ('TYPEII', 'Indicates PHA Type II file format')
    spectrum.header['HDUVERS'] = ('1.2.1', 'Version of HDUCLAS1 format in use')
    spectrum.header['CHANTYPE'] = (chantype, 'Channel type')
    spectrum.header['DETCHANS'] = (len(channel), 'Total number of channels in each rate')

    fits.HDUList([primary, ebounds, spectrum]).writeto(specfile, overwrite=True)