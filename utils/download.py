# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:10:37 2022

@author: xuewc
"""

import os
from .time import get_YMDh


def download_tte(utc0, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dets = [
        'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb',
        'b0', 'b1'
    ]
    Y, M, D, h = get_YMDh(utc0)
    tstr = f'{Y[-2:]}{M}{D}_{h}'
    url = f'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{Y}/{M}/{D}'

    poshist_url =  f'{url}/current/glg_poshist_all_{tstr[:-3]}_v00.fit'
    os.system(f'curl -O {poshist_url} --output-dir {output_dir}')

    for det in dets:
        url_d = f'{url}/current/glg_tte_{det}_{tstr}z_v00.fit.gz'
        os.system(f'curl -O {url_d} --output-dir {output_dir}')
