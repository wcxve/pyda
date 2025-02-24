# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

import glob
import os

# fix: Unable to redirect prompts to the /dev/tty (at headas_stdio.c:152)
os.environ["HEADASNOQUERY"] = "False"

import numpy as np

from astropy.io import fits


def get_pattern_matched_file(pattern, pattern2=None):
    files = glob.glob(pattern)
    if len(files):
        return sorted(files)[-1]
    else:
        if pattern2 is not None:
            files = glob.glob(pattern2)
            if len(files):
                return sorted(files)[-1]
            else:
                raise ValueError(
                    f'No files matched pattern:\n{pattern}\nor{pattern2}'
                )
        else:
            raise ValueError(
                f'No files matched pattern:\n{pattern}'
            )


def specify_hxmt_gti(infile, intervals, outfile):
    intervals = np.reshape(intervals, newshape=(-1, 2))

    with fits.open(infile) as f:
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header = f['PRIMARY'].header

        gti0_hdu = fits.BinTableHDU.from_columns(
            columns=fits.ColDefs([
                fits.Column('START', '1D', 's', array=intervals[:, 0]),
                fits.Column('STOP', '1D', 's', array=intervals[:, 1])
            ]),
            header=f['GTI0'].header,
            name='GTI0'
        )

        gtidesc_hdu = fits.BinTableHDU()
        gtidesc_hdu.name = 'GTIDesc'
        gtidesc_hdu.header = f['GTIDesc'].header
        gtidesc_hdu.header.remove(
            keyword='HISTORY', ignore_missing=True, remove_all=True
        )
        gtidesc_hdu.header.add_history(
            f'GTI specified by user: {np.squeeze(intervals).tolist()}'
        )
        gtidesc_hdu.data = f['GTIDesc'].data

    hdul = fits.HDUList([primary_hdu, gti0_hdu, gtidesc_hdu])
    hdul.writeto(outfile, overwrite=True)


def le_reduction(
    obsid,
    out_path,
    tstart,
    tstop,
    binsize,
    tstart_src,
    tstop_src,
    bkg_intervals=None,
    emin=1,
    emax=10,
    det_expr='0,2-4,6-10,12,14,20,22-26,28,30,32,34-36,38-42,44,46,52,54-58,60-62,64,66-68,70-74,76,78,84,86,88-90,92-94',
    gti_expr='ELV>10&&DYE_ELV>30&&COR>8&&SAA_FLAG==0&&T_SAA>=300&&TN_SAA>=300&&ANG_DIST<=0.04',
    overwrite=True,
    overwrite_pi=False,
    overwrite_recon=False,
    dirname=None
):
    """
    Insight-HXMT/LE Data reduction.

    Parameters
    ----------
    obsid : str
        ObsID of HXMT observation, e.g., P051435700205.
    out_path : str
        Output path.
    tstart : float, or int
        Start time of screen file in MET format.
    tstop : float, or int
        Stop time of screen file in MET format.
    binsize : float or int
        Binsize of light curve, used to extract exposure info in that scale.
    tstart_src : float or int
        Start time of source region in MET format.
    tstop_src : float or int
        Stop time of source region in MET format.
    bkg_intervals : 1-D array_like
        Background intervals in MET format. e.g., [t0-11, t0-1, t0+1, t0+11].
    emin : float or int
        Minimum energy to consider in light curve.
    emax : float or int
        Maximum energy to consider in light curve.
    det_expr : str, optional
        Detector selection criterion.
    gti_expr : str, optional
        Good time interval criterion.
    overwrite : bool, optional
        Whether to overwrite files, excluding PI and RECON file.
    overwrite_pi : bool, optional
        Whether to overwrite PI file.
    overwrite_recon : bool, optional
        Whether to overwrite RECON file.
    dirname : str, or None, optional
        Output directory of 2L product. When None, it is given as
        f'{tstart}-{tstop}'.

    Returns
    -------
    flag : int
        1 when success.

    """
    if tstart_src < tstart:
        raise ValueError('`tstart_src` is smaller than `tstart`')

    if tstop_src > tstop:
        raise ValueError('`tstop_src` is larger than `tstop`')

    if emin < 0.1:
        raise ValueError('`emin` is smaller than 0.1 keV')

    if emax > 13.1:
        raise ValueError('`emax` is larger than 13.1 keV')

    if bkg_intervals is not None:
        if len(bkg_intervals) % 2 != 0:
            raise ValueError('length of `bkg_intervals` is not multiple of 2')
        if np.any(np.diff(bkg_intervals) < 0.0):
            raise ValueError('`bkg_intervals` is not incremental in time')
        if bkg_intervals[0] < tstart:
            raise ValueError('`bkg_intervals[0]` is smaller than `tstart`')
        if bkg_intervals[-1] > tstop:
            raise ValueError('`bkg_intervals[-1]` is larger than `tstop`')

    clobber_pi = 'yes' if overwrite_pi else 'no'
    clobber_recon = 'yes' if overwrite_recon else 'no'
    clobber = 'yes' if overwrite else 'no'

    p1 = f'/hxmt/work/HXMT-DATA/1L/A{obsid[1:3]}/{obsid[:-5]}/{obsid[:-2]}'
    p2 = f'{p1}/{obsid}-????????-??-??'

    att_pattern      = f'{p2}/ACS/HXMT_{obsid}_Att_FFFFFF_V?_L1P.FITS'
    ehk_pattern      = f'{p2}/AUX/HXMT_{obsid}_EHK_FFFFFF_V?_L1P.FITS'
    att_pattern2     = f'{p1}/ACS/HXMT_{obsid[:-2]}_Att_FFFFFF_V?_L1P.FITS'
    ehk_pattern2     = f'{p1}/AUX/HXMT_{obsid[:-2]}_EHK_FFFFFF_V?_L1P.FITS'
    evt_pattern      = f'{p2}/LE/HXMT_{obsid}_LE-Evt_FFFFFF_V?_L1P.FITS'
    temp_pattern     = f'{p2}/LE/HXMT_{obsid}_LE-TH_FFFFFF_V?_L1P.FITS'
    instatus_pattern = f'{p2}/LE/HXMT_{obsid}_LE-InsStat_FFFFFF_V?_L1P.FITS'

    att_file      = get_pattern_matched_file(att_pattern, att_pattern2)
    ehk_file      = get_pattern_matched_file(ehk_pattern, ehk_pattern2)
    evt_file      = get_pattern_matched_file(evt_pattern)
    temp_file     = get_pattern_matched_file(temp_pattern)
    instatus_file = get_pattern_matched_file(instatus_pattern)

    if not os.path.exists(f'{out_path}/{obsid}'):
        os.system(f'mkdir {out_path}/{obsid}')
    if not os.path.exists(f'{out_path}/{obsid}/LE'):
        os.system(f'mkdir {out_path}/{obsid}/LE')

    prefix1    = f'{out_path}/{obsid}/LE/{obsid}_LE'
    pi_file    = f'{prefix1}_PI.fits'
    pedestal   = f'{prefix1}_PEDESTAL.fits'
    recon_file = f'{prefix1}_RECON.fits'
    gti_tmp    = f'{prefix1}_GTI.tmp'
    gti_file   = f'{prefix1}_GTI.fits'

    os.system(
        f'lepical '
        f'evtfile="{evt_file}" '
        f'tempfile="{temp_file}" '
        f'outfile="{pi_file}" '
        f'pedestalfile="{pedestal}" '
        f'clobber={clobber_pi}'
    )

    os.system(
        f'lerecon '
        f'evtfile="{pi_file}" '
        f'outfile="{recon_file}" '
        f'instatusfile="{instatus_file}" '
        f'clobber={clobber_recon}'
    )

    os.system(
        f'legtigen '
        f'evtfile="NONE" '
        f'instatusfile="{instatus_file}" '
        f'tempfile="{temp_file}" '
        f'ehkfile="{ehk_file}" '
        f'outfile="{gti_tmp}" '
        f'defaultexpr=NONE '
        f'expr="{gti_expr}" '
        f'clobber={clobber}'
    )
    os.system(
        f'legticorr "{recon_file}" "{gti_tmp}" "{gti_file}"'
    )

    # produce 2L Data
    if dirname is not None:
        path_2L = f'{out_path}/{obsid}/{dirname}'
    else:
        path_2L = f'{out_path}/{obsid}/{tstart}-{tstop}'
    if not os.path.exists(path_2L):
        os.system(f'mkdir {path_2L}')

    prefix2   = f'{path_2L}/LE'
    gti_user  = f'{prefix2}_GTI.fits'
    evt_data  = f'{prefix2}_EVT.fits'
    lc_file   = f'{prefix2}'
    spec_file = f'{prefix2}_TOTAL'
    rsp_file  = f'{prefix2}.rsp'

    # generate screened event data given DetID and time interval of interest
    specify_hxmt_gti(gti_file, (tstart, tstop), gti_user)
    os.system(
        f'lescreen '
        f'evtfile="{recon_file}" '
        f'gtifile="{gti_user}" '
        f'outfile="{evt_data}" '
        f'userdetid="{det_expr}" '
        f'eventtype=1 '
        f'starttime={tstart} stoptime={tstop} minPI=0 maxPI=1535 '
        f'clobber={clobber}'
    )

    # generate spectrum given DetID and time interval of interest
    os.system(
        f'lespecgen '
        f'evtfile="{evt_data}" '
        f'outfile="{spec_file}" '
        f'eventtype=1 '
        f'userdetid="{det_expr}" '
        f'starttime={tstart_src} stoptime={tstop_src} minPI=0 maxPI=1535 '
        f'clobber={clobber}'
    )

    # generate background spectrum
    if bkg_intervals is not None:
        gti_bkg  = f'{prefix2}_BKG_GTI.fits'
        bkg_data = f'{prefix2}_BKG_EVT.fits'
        bkg_spec = f'{prefix2}_BKG'
        specify_hxmt_gti(gti_file, bkg_intervals, gti_bkg)
        os.system(
            f'lescreen '
            f'evtfile="{recon_file}" '
            f'gtifile="{gti_bkg}" '
            f'outfile="{bkg_data}" '
            f'userdetid="{det_expr}" '
            f'eventtype=1 '
            f'starttime=0 stoptime=0 minPI=0 maxPI=1535 '
            f'clobber={clobber}'
        )
        os.system(
            f'lespecgen '
            f'evtfile="{bkg_data}" '
            f'outfile="{bkg_spec}" '
            f'eventtype=1 '
            f'userdetid="{det_expr}" '
            f'starttime=0 stoptime=0 minPI=0 maxPI=1535 '
            f'clobber={clobber}'
        )

    # generate detector response file
    phafile = get_pattern_matched_file(f'{spec_file}_g0.pha')
    os.system(
        f'lerspgen '
        f'phafile="{phafile}" '
        f'outfile="{rsp_file}" '
        f'attfile="{att_file}" '
        f'tempfile="{temp_file}" '
        f'ra="-1" dec="-91" '
        f'clobber={clobber}'
    )

    # generate light curve given DetID and time, to get exposure information
    with fits.open(rsp_file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        emask = (emin <= ebounds['E_MIN']) & (ebounds['E_MAX'] <= emax)
        minPI = ebounds[emask]['CHANNEL'].min()
        maxPI = ebounds[emask]['CHANNEL'].max()
    os.system(
        f'lelcgen '
        f'evtfile="{evt_data}" '
        f'outfile="{lc_file}" '
        f'userdetid="{det_expr}" '
        f'binsize={binsize} '
        f'starttime={tstart} stoptime={tstop} '
        f'minPI={minPI} maxPI={maxPI} '
        f'eventtype=1 '
        f'clobber={clobber}'
    )

    return 1


def me_reduction(
    obsid,
    out_path,
    tstart,
    tstop,
    binsize,
    tstart_src,
    tstop_src,
    bkg_intervals=None,
    emin=10,
    emax=35,
    det_expr='0-7,11-25,29-43,47-53',
    gti_expr='ELV>10&&COR>8&&SAA_FLAG==0&&T_SAA>=300&&TN_SAA>=300&&ANG_DIST<=0.04',
    overwrite=True,
    overwrite_pi=False,
    overwrite_grade=False,
    dirname=None
):
    """
    Insight-HXMT/ME Data reduction.

    Parameters
    ----------
    obsid : str
        ObsID of HXMT observation, e.g., P051435700205.
    out_path : str
        Output path.
    tstart : float, or int
        Start time of screen file in MET format.
    tstop : float, or int
        Stop time of screen file in MET format.
    binsize : float or int
        Binsize of light curve, used to extract exposure info in that scale.
    tstart_src : float or int
        Start time of source region in MET format.
    tstop_src : float or int
        Stop time of source region in MET format.
    bkg_intervals : 1-D array_like
        Background intervals in MET format. e.g., [t0-11, t0-1, t0+1, t0+11].
    emin : float or int
        Minimum energy to consider in light curve.
    emax : float or int
        Maximum energy to consider in light curve.
    det_expr : str, optional
        Detector selection criterion.
    gti_expr : str, optional
        Good time interval criterion.
    overwrite : bool, optional
        Whether to overwrite files, excluding PI and GRADE file.
    overwrite_pi : bool, optional
        Whether to overwrite PI file.
    overwrite_grade : bool, optional
        Whether to overwrite GRADE file.
    dirname : str, or None, optional
        Output directory of 2L product. When None, it is given as
        f'{tstart}-{tstop}'.

    Returns
    -------
    flag : int
        1 when success.

    """
    if tstart_src < tstart:
        raise ValueError('`tstart_src` is smaller than `tstart`')

    if tstop_src > tstop:
        raise ValueError('`tstop_src` is larger than `tstop`')

    if emin < 3:
        raise ValueError('`emin` is smaller than 3 keV')

    if emax > 63:
        raise ValueError('`emax` is larger than 63 keV')

    if bkg_intervals is not None:
        if len(bkg_intervals) % 2 != 0:
            raise ValueError('length of `bkg_intervals` is not multiple of 2')
        if np.any(np.diff(bkg_intervals) < 0.0):
            raise ValueError('`bkg_intervals` is not incremental in time')
        if bkg_intervals[0] < tstart:
            raise ValueError('`bkg_intervals[0]` is smaller than `tstart`')
        if bkg_intervals[-1] > tstop:
            raise ValueError('`bkg_intervals[-1]` is larger than `tstop`')

    clobber_pi = 'yes' if overwrite_pi else 'no'
    clobber_grade = 'yes' if overwrite_grade else 'no'
    clobber = 'yes' if overwrite else 'no'

    p1 = f'/hxmt/work/HXMT-DATA/1L/A{obsid[1:3]}/{obsid[:-5]}/{obsid[:-2]}'
    p2 = f'{p1}/{obsid}-????????-??-??'

    att_pattern  = f'{p2}/ACS/HXMT_{obsid}_Att_FFFFFF_V?_L1P.FITS'
    ehk_pattern  = f'{p2}/AUX/HXMT_{obsid}_EHK_FFFFFF_V?_L1P.FITS'
    att_pattern2 = f'{p1}/ACS/HXMT_{obsid[:-2]}_Att_FFFFFF_V?_L1P.FITS'
    ehk_pattern2 = f'{p1}/AUX/HXMT_{obsid[:-2]}_EHK_FFFFFF_V?_L1P.FITS'
    evt_pattern  = f'{p2}/ME/HXMT_{obsid}_ME-Evt_FFFFFF_V?_L1P.FITS'
    temp_pattern = f'{p2}/ME/HXMT_{obsid}_ME-TH_FFFFFF_V?_L1P.FITS'

    att_file  = get_pattern_matched_file(att_pattern, att_pattern2)
    ehk_file  = get_pattern_matched_file(ehk_pattern, ehk_pattern2)
    evt_file  = get_pattern_matched_file(evt_pattern)
    temp_file = get_pattern_matched_file(temp_pattern)

    if not os.path.exists(f'{out_path}/{obsid}'):
        os.system(f'mkdir {out_path}/{obsid}')
    if not os.path.exists(f'{out_path}/{obsid}/ME'):
        os.system(f'mkdir {out_path}/{obsid}/ME')

    prefix1    = f'{out_path}/{obsid}/ME/{obsid}_ME'
    pi_file    = f'{prefix1}_PI.fits'
    grade_file = f'{prefix1}_GRADE.fits'
    dtime_file = f'{prefix1}_DTIME.fits'
    gti_tmp    = f'{prefix1}_GTI.tmp'
    gti_file   = f'{prefix1}_GTI.fits'
    det_stat   = f'{prefix1}_DSTAT.fits'

    os.system(
        f'mepical '
        f'evtfile="{evt_file}" '
        f'tempfile="{temp_file}" '
        f'outfile="{pi_file}" '
        f'clobber={clobber_pi}'
    )

    os.system(
        f'megrade '
        f'evtfile="{pi_file}" '
        f'outfile="{grade_file}" '
        f'deadfile={dtime_file} '
        f'binsize=0.001 '
        f'clobber={clobber_grade}'
    )

    os.system(
        f'megtigen '
        f'tempfile="{temp_file}" '
        f'ehkfile="{ehk_file}" '
        f'outfile="{gti_tmp}" '
        f'defaultexpr=NONE '
        f'expr="{gti_expr}" '
        f'clobber={clobber}'
    )
    os.system(
        f'megticorr "{grade_file}" "{gti_tmp}" "{gti_file}" '
        f'$HEADAS/refdata/medetectorstatus.fits "{det_stat}"'
    )

    # produce 2L Data
    if dirname is not None:
        path_2L = f'{out_path}/{obsid}/{dirname}'
    else:
        path_2L = f'{out_path}/{obsid}/{tstart}-{tstop}'
    if not os.path.exists(path_2L):
        os.system(f'mkdir {path_2L}')

    prefix2   = f'{path_2L}/ME'
    gti_user  = f'{prefix2}_GTI.fits'
    evt_data  = f'{prefix2}_EVT.fits'
    lc_file   = f'{prefix2}'
    spec_file = f'{prefix2}_TOTAL'
    rsp_file  = f'{prefix2}.rsp'

    # generate screened event data given DetID and time interval of interest
    specify_hxmt_gti(gti_file, (tstart, tstop), gti_user)
    os.system(
        f'mescreen '
        f'evtfile="{grade_file}" '
        f'gtifile="{gti_user}" '
        f'outfile="{evt_data}" '
        f'baddetfile="{det_stat}" '
        f'userdetid="{det_expr}" '
        f'starttime={tstart} stoptime={tstop} minPI=0 maxPI=1023 '
        f'clobber={clobber}'
    )

    # generate spectrum given DetID and time interval of interest
    os.system(
        f'mespecgen '
        f'evtfile="{evt_data}" '
        f'outfile="{spec_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'starttime={tstart_src} stoptime={tstop_src} minPI=0 maxPI=1023 '
        f'clobber={clobber}'
    )

    # generate background spectrum
    if bkg_intervals is not None:
        gti_bkg  = f'{prefix2}_BKG_GTI.fits'
        bkg_data = f'{prefix2}_BKG_EVT.fits'
        bkg_spec = f'{prefix2}_BKG'
        specify_hxmt_gti(gti_file, bkg_intervals, gti_bkg)
        os.system(
            f'mescreen '
            f'evtfile="{grade_file}" '
            f'gtifile="{gti_bkg}" '
            f'outfile="{bkg_data}" '
            f'baddetfile="{det_stat}" '
            f'userdetid="{det_expr}" '
            f'starttime=0 stoptime=0 minPI=0 maxPI=1023 '
            f'clobber={clobber}'
        )
        os.system(
            f'mespecgen '
            f'evtfile="{bkg_data}" '
            f'outfile="{bkg_spec}" '
            f'deadfile="{dtime_file}" '
            f'userdetid="{det_expr}" '
            f'starttime=0 stoptime=0 minPI=0 maxPI=1023 '
            f'clobber={clobber}'
        )

    # generate detector response file
    phafile = get_pattern_matched_file(f'{spec_file}_g0.pha')
    os.system(
        f'merspgen '
        f'phafile="{phafile}" '
        f'outfile="{rsp_file}" '
        f'attfile="{att_file}" '
        f'ra="-1" dec="-91" '
        f'clobber={clobber}'
    )

    # generate light curve given DetID and time, to get exposure information
    with fits.open(rsp_file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        emask = (emin <= ebounds['E_MIN']) & (ebounds['E_MAX'] <= emax)
        minPI = ebounds[emask]['CHANNEL'].min()
        maxPI = ebounds[emask]['CHANNEL'].max()
    os.system(
        f'melcgen '
        f'evtfile="{evt_data}" '
        f'outfile="{lc_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'binsize={binsize} '
        f'starttime={tstart} stoptime={tstop} '
        f'minPI={minPI} maxPI={maxPI} '
        f'deadcorr=no '
        f'clobber={clobber}'
    )

    return 1


def he_reduction(
    obsid,
    out_path,
    tstart,
    tstop,
    binsize,
    tstart_src,
    tstop_src,
    bkg_intervals=None,
    emin=28,
    emax=250,
    det_expr='0-15,17',
    gti_expr='ELV>10&&COR>8&&SAA_FLAG==0&&T_SAA>=300&&TN_SAA>=300&&ANG_DIST<=0.04',
    pm_expr='Cnt_PM_0<50&&Cnt_PM_1<50&&Cnt_PM_2<50',
    overwrite=True,
    overwrite_pi=False,
    dirname=None
):
    """
    Insight-HXMT/HE Data reduction.

    Parameters
    ----------
    obsid : str
        ObsID of HXMT observation, e.g., P051435700205.
    out_path : str
        Output path.
    tstart : float, or int
        Start time of screen file in MET format.
    tstop : float, or int
        Stop time of screen file in MET format.
    binsize : float or int
        Binsize of light curve, used to extract exposure info in that scale.
    tstart_src : float or int
        Start time of source region in MET format.
    tstop_src : float or int
        Stop time of source region in MET format.
    bkg_intervals : 1-D array_like
        Background intervals in MET format. e.g., [t0-11, t0-1, t0+1, t0+11].
    emin : float or int
        Minimum energy to consider in light curve.
    emax : float or int
        Maximum energy to consider in light curve.
    det_expr : str, optional
        Detector selection criterion.
    gti_expr : str, optional
        Good time interval criterion.
    pm_expr : str, optional
        Particle monitor criterion for GTI.
    overwrite : bool, optional
        Whether to overwrite files, excluding PI file.
    overwrite_pi : bool, optional
        Whether to overwrite PI file.
    dirname : str, or None, optional
        Output directory of 2L product. When None, it is given as
        f'{tstart}-{tstop}'.

    Returns
    -------
    flag : int
        1 when success.

    """
    if tstart_src < tstart:
        raise ValueError('`tstart_src` is smaller than `tstart`')

    if tstop_src > tstop:
        raise ValueError('`tstop_src` is larger than `tstop`')

    if emin < 15:
        raise ValueError('`emin` is smaller than 15 keV')

    if emax > 385:
        raise ValueError('`emax` is larger than 385 keV')

    if bkg_intervals is not None:
        if len(bkg_intervals) % 2 != 0:
            raise ValueError('length of `bkg_intervals` is not multiple of 2')
        if np.any(np.diff(bkg_intervals) < 0.0):
            raise ValueError('`bkg_intervals` is not incremental in time')
        if bkg_intervals[0] < tstart:
            raise ValueError('`bkg_intervals[0]` is smaller than `tstart`')
        if bkg_intervals[-1] > tstop:
            raise ValueError('`bkg_intervals[-1]` is larger than `tstop`')

    clobber_pi = 'yes' if overwrite_pi else 'no'
    clobber = 'yes' if overwrite else 'no'

    p1 = f'/hxmt/work/HXMT-DATA/1L/A{obsid[1:3]}/{obsid[:-5]}/{obsid[:-2]}'
    p2 = f'{p1}/{obsid}-????????-??-??'

    att_pattern   = f'{p2}/ACS/HXMT_{obsid}_Att_FFFFFF_V?_L1P.FITS'
    ehk_pattern   = f'{p2}/AUX/HXMT_{obsid}_EHK_FFFFFF_V?_L1P.FITS'
    att_pattern2  = f'{p1}/ACS/HXMT_{obsid[:-2]}_Att_FFFFFF_V?_L1P.FITS'
    ehk_pattern2  = f'{p1}/AUX/HXMT_{obsid[:-2]}_EHK_FFFFFF_V?_L1P.FITS'
    dtime_pattern = f'{p2}/HE/HXMT_{obsid}_HE-DTime_FFFFFF_V?_L1P.FITS'
    evt_pattern   = f'{p2}/HE/HXMT_{obsid}_HE-Evt_FFFFFF_V?_L1P.FITS'
    hv_pattern    = f'{p2}/HE/HXMT_{obsid}_HE-HV_FFFFFF_V?_L1P.FITS'
    temp_pattern  = f'{p2}/HE/HXMT_{obsid}_HE-TH_FFFFFF_V?_L1P.FITS'
    pm_pattern    = f'{p2}/HE/HXMT_{obsid}_HE-PM_FFFFFF_V?_L1P.FITS'

    att_file   = get_pattern_matched_file(att_pattern, att_pattern2)
    ehk_file   = get_pattern_matched_file(ehk_pattern, ehk_pattern2)
    dtime_file = get_pattern_matched_file(dtime_pattern)
    evt_file   = get_pattern_matched_file(evt_pattern)
    hv_file    = get_pattern_matched_file(hv_pattern)
    temp_file  = get_pattern_matched_file(temp_pattern)
    pm_file    = get_pattern_matched_file(pm_pattern)


    if not os.path.exists(f'{out_path}/{obsid}'):
        os.system(f'mkdir {out_path}/{obsid}')
    if not os.path.exists(f'{out_path}/{obsid}/HE'):
        os.system(f'mkdir {out_path}/{obsid}/HE')

    prefix1  = f'{out_path}/{obsid}/HE/{obsid}_HE'
    pi_file  = f'{prefix1}_PI.fits'
    gti_file = f'{prefix1}_GTI.fits'

    os.system(
        f'hepical '
        f'evtfile="{evt_file}" '
        f'outfile="{pi_file}" '
        f'clobber={clobber_pi}'
    )

    os.system(
        f'hegtigen '
        f'hvfile="{hv_file}" '
        f'tempfile="{temp_file}" '
        f'pmfile="{pm_file}" '
        f'outfile="{gti_file}" '
        f'ehkfile="{ehk_file}" '
        f'defaultexpr=NONE '
        f'expr="{gti_expr}" '
        f'pmexpr="{pm_expr}" '
        f'clobber={clobber}'
    )

    # produce 2L Data
    if dirname is not None:
        path_2L = f'{out_path}/{obsid}/{dirname}'
    else:
        path_2L = f'{out_path}/{obsid}/{tstart}-{tstop}'
    if not os.path.exists(path_2L):
        os.system(f'mkdir {path_2L}')

    prefix2   = f'{path_2L}/HE'
    gti_user  = f'{prefix2}_GTI.fits'
    evt_data  = f'{prefix2}_EVT.fits'
    lc_file   = f'{prefix2}'
    spec_file = f'{prefix2}_TOTAL'
    rsp_file  = f'{prefix2}.rsp'

    # generate screened event data given DetID and time interval of interest
    specify_hxmt_gti(gti_file, (tstart, tstop), gti_user)
    os.system(
        f'hescreen '
        f'evtfile="{pi_file}" '
        f'gtifile="{gti_user}" '
        f'outfile="{evt_data}" '
        f'userdetid="{det_expr}" '
        f'eventtype=1 '
        f'anticoincidence=yes '
        f'starttime={tstart} stoptime={tstop} minPI=0 maxPI=255 '
        f'clobber={clobber}'
    )

    # generate spectrum given DetID and time interval of interest
    os.system(
        f'hespecgen '
        f'evtfile="{evt_data}" '
        f'outfile="{spec_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'eventtype=1 '
        f'starttime={tstart_src} stoptime={tstop_src} minPI=0 maxPI=255 '
        f'clobber={clobber}'
    )

    # generate background spectrum
    if bkg_intervals is not None:
        gti_bkg  = f'{prefix2}_BKG_GTI.fits'
        bkg_data = f'{prefix2}_BKG_EVT.fits'
        bkg_spec = f'{prefix2}_BKG'
        specify_hxmt_gti(gti_file, bkg_intervals, gti_bkg)
        os.system(
            f'hescreen '
            f'evtfile="{pi_file}" '
            f'gtifile="{gti_bkg}" '
            f'outfile="{bkg_data}" '
            f'userdetid="{det_expr}" '
            f'eventtype=1 '
            f'anticoincidence=yes '
            f'starttime=0 stoptime=0 minPI=0 maxPI=255 '
            f'clobber={clobber}'
        )
        os.system(
            f'hespecgen '
            f'evtfile="{bkg_data}" '
            f'outfile="{bkg_spec}" '
            f'deadfile="{dtime_file}" '
            f'userdetid="{det_expr}" '
            f'eventtype=1 '
            f'starttime=0 stoptime=0 minPI=0 maxPI=1023 '
            f'clobber={clobber}'
        )

    # generate detector response file
    phafile = get_pattern_matched_file(f'{spec_file}_g0.pha')
    os.system(
        f'herspgen '
        f'phafile="{phafile}" '
        f'outfile="{rsp_file}" '
        f'attfile="{att_file}" '
        f'ra="-1" dec="-91" '
        f'clobber={clobber}'
    )

    # generate light curve given DetID and time, to get exposure information
    with fits.open(rsp_file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        emask = (emin <= ebounds['E_MIN']) & (ebounds['E_MAX'] <= emax)
        minPI = ebounds[emask]['CHANNEL'].min()
        maxPI = ebounds[emask]['CHANNEL'].max()
    os.system(
        f'helcgen '
        f'evtfile="{evt_data}" '
        f'outfile="{lc_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'binsize={binsize} '
        f'starttime={tstart} stoptime={tstop} '
        f'minPI={minPI} maxPI={maxPI} '
        f'deadcorr=no '
        f'clobber={clobber}'
    )

    return 1
