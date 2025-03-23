#!/usr/bin/env python

import numpy as np
import pandas as pd
from pyscf import gto, scf, lib
import sys
import argparse
from pathlib import Path

def extract_parameters(filename):
    # Remove the prefix and suffix to isolate the parameters
    base_name = filename.replace('stacked_benzene_hexafluorobenzene_', '').replace('.xyz', '')
    
    # Split the remaining string by underscore and extract the parameters
    parts = base_name.split('_')
    
    num_rings = int(parts[0])
    distance = float(parts[2])
    displacement = float(parts[4])
    
    return num_rings, distance, displacement

def perform_dft_calculation(xyz_filename):
    mol = gto.M(atom=str(xyz_filename), basis="6-31G*")
    mf = scf.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    n = mol.nelec[0] + mol.nelec[1]  # Total number of electrons
    homo = mf.mo_energy[int(n/2) - 1]
    lumo = mf.mo_energy[int(n/2)]
    homo_eV = homo * 27.2114
    lumo_eV = lumo * 27.2114
    gap_eV = lumo_eV - homo_eV
    dft_energy = mf.e_tot  # Extract the total DFT energy
    return homo_eV, lumo_eV, gap_eV, dft_energy

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xyz_file', help='path to XYZ file', required=True)
    parser.add_argument('-n', '--num_threads', help='number of threads for PySCF', default=1)
    args = parser.parse_args()

    # Set the number of threads
    lib.num_threads(args.num_threads)

    results = []

    print(f"Running PySCF on {args.xyz_file} with {args.num_threads} threads")
    xyz_filename = Path( args.xyz_file )
    print("output will be written to:", f'./dft_results_tshape_parallel_benhex-{xyz_filename.stem}.csv' )

    try:
        homo, lumo, gap, dft_energy = perform_dft_calculation(xyz_filename)
        if homo is not None and lumo is not None and gap is not None:
            print(f"File: {xyz_filename}")
            print(f"HOMO: {homo:.6f} eV")
            print(f"LUMO: {lumo:.6f} eV")
            print(f"HOMO-LUMO Gap: {gap:.6f} eV")
            print(f"DFT Energy: {dft_energy:.6f} Hartree\n")

            # Extract parameters from the filename
            num_rings, distance, displacement = extract_parameters(xyz_filename.stem)

            results.append({
                'File': xyz_filename.name,
                'HOMO (eV)': homo,
                'LUMO (eV)': lumo,
                'Gap (eV)': gap,
                'DFT Energy (Hartree)': dft_energy,
                'Num Rings': num_rings,
                'Distance': distance,
                'Displacement': displacement
            })
            
        else:
            print(f"Calculation failed for file: {xyz_filename}\n")
    except Exception as e:
        print(f"Error processing file {xyz_filename}: {str(e)}\n")

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    df.to_csv(f'./dft_results_tshape_parallel_benhex-{xyz_filename.stem}.csv', index=False)

if __name__ == "__main__":
    main()
    
