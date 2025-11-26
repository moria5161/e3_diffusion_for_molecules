import numpy as np
import torch
import requests

import logging
import os
import urllib

from os.path import join as join
import urllib.request

from qm9.data.prepare.process import process_xyz_files, process_xyz_gdb9
from qm9.data.prepare.utils import download_data, is_int, cleanup_file


def download_with_headers(url, filename):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://springernature.figshare.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    # 增加超时设置，防止卡死
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def download_dataset_qm9(datadir, dataname, splits=None, calculate_thermo=True, exclude=True, cleanup=True):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    gdb9dir = join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(gdb9dir, exist_ok=True)

    logging.info(
        'Downloading and processing GDB9 dataset. Output will be in directory: {}.'.format(gdb9dir))

    logging.info('Beginning download of GDB9 dataset!')
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    
    # [修改点 1] 检查文件是否存在，如果存在则跳过下载
    if os.path.exists(gdb9_tar_data):
        logging.info('GDB9 tar file already exists at {}, skipping download.'.format(gdb9_tar_data))
    else:
        logging.info('Downloading GDB9 tar file...')
        download_with_headers(gdb9_url_data, gdb9_tar_data)
        logging.info('GDB9 dataset downloaded successfully!')

    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_gdb9(gdb9dir, cleanup)

    # Process GDB9 dataset, and return dictionary of splits
    gdb9_data = {}
    for split, split_idx in splits.items():
        gdb9_data[split] = process_xyz_files(
            gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx, stack=True)

    # Subtract thermochemical energy if desired.
    if calculate_thermo:
        # Download thermochemical energy from GDB9 dataset, and then process it into a dictionary
        therm_energy = get_thermo_dict(gdb9dir, cleanup)

        # For each of train/validation/test split, add the thermochemical energy
        for split_idx, split_data in gdb9_data.items():
            gdb9_data[split_idx] = add_thermo_targets(split_data, therm_energy)

    # Save processed GDB9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in gdb9_data.items():
        savedir = join(gdb9dir, split+'.npz')
        np.savez_compressed(savedir, **data)

    logging.info('Processing/saving complete!')


def gen_splits_gdb9(gdb9dir, cleanup=True):
    """
    Generate GDB9 training/validation/test splits used.
    """
    logging.info('Splits were not specified! Automatically generating.')
    gdb9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
    gdb9_txt_excluded = join(gdb9dir, 'uncharacterized.txt')
    
    # [修改点 2] 检查排除列表文件是否存在
    if os.path.exists(gdb9_txt_excluded):
        logging.info('Excluded molecules file already exists, skipping download.')
    else:
        logging.info('Downloading excluded molecules file...')
        download_with_headers(gdb9_url_excluded, gdb9_txt_excluded)

    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0]
                            for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
        len(excluded_idxs))

    # Now, create a list of indices
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(
        sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    splits = {'train': train, 'valid': valid, 'test': test}

    # Cleanup
    cleanup_file(gdb9_txt_excluded, cleanup)

    return splits


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.
    """
    # Download thermochemical energy
    logging.info('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')

    # [修改点 3] 检查原子能量参考文件是否存在
    if os.path.exists(gdb9_txt_thermo):
        logging.info('Thermochemical energy file already exists, skipping download.')
    else:
        logging.info('Downloading thermochemical energy file...')
        download_with_headers(gdb9_url_thermo, gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy


def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo

    return data


def get_unique_charges(charges):
    """
    Get count of each charge for each molecule.
    """
    # Create a dictionary of charges
    # [修改点] 将 np.int 改为 int
    charge_counts = {z: np.zeros(len(charges), dtype=int)
                     for z in np.unique(charges)}
    print(charge_counts.keys())

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[z][idx] = num_z

    return charge_counts