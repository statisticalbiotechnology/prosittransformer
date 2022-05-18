import sys
from pyteomics import mzml, mass
import numpy as np
import pandas as pd

#import spectrum_utils.spectrum as sus
#import spectrum_utils.plot as sup
import matplotlib.pyplot as plt

import torch 
from tape import TAPETokenizer
from tape import ProteinBertForValuePredictionFragmentationProsit
from prosittransformer.DataHandler import pad_sequences
from prosittransformer.utils import cleanTapeOutput 


mod_masses = mass.std_aa_mass.copy()
mod_masses['C'] = 57.02146 ## Adjust for carbamidomethylated cysteines -static mod-

pt_model_path = "./torch_model"

def getPrositTransformerModel():
    model = ProteinBertForValuePredictionFragmentationProsit.from_pretrained(pt_model_path)
    model = model.to(torch.device('cuda:0'))
    return model

def get_precursor_charge_onehot(charges, considered_charges = range(1,7)):
    if isinstance(charges, int):
        charges = [charges]
    array = np.zeros([len(charges), max(considered_charges)], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, precursor_charge - 1] = 1
    return array


def TokenizePeptides(peptides, tokenizer = TAPETokenizer()):
    if isinstance(peptides, str):
        peptides = [peptides]
    input_ids = pad_sequences([tokenizer.encode(p) for p in peptides])
    return input_ids, np.ones_like(input_ids)



def predictSpectrum(peptide, charge, ce = 20):
    p_charges = get_precursor_charge_onehot(charge)
    p_ces = np.hstack([ce])
    input_ids, input_mask = TokenizePeptides(peptide)
    model = getPrositTransformerModel()

def predictSpectra(peptides, charges, ces = 20):
    p_charges = get_precursor_charge_onehot(charges)
    p_ces = np.hstack(ces)
    input_ids, input_mask = TokenizePeptides(peptides)
    model = getPrositTransformerModel()
    prd_peaks = []
    prd_elapsed = 0
    prd_batch = 200
    while prd_elapsed < len(peptides): ## Predict until all valid PSMs are accounted for
        if (prd_elapsed + prd_batch) > len(peptides): ## Truncate last prediction to match #PSMS
            prd_batch = len(peptides) - prd_elapsed

        targets_data = { ## Load a batch of PSM CEs, charges, and peptide identifiers
           'collision_energy' : torch.FloatTensor(p_ces[prd_elapsed: (prd_elapsed+prd_batch)].astype(np.float32)),
           'charge': torch.FloatTensor(p_charges[prd_elapsed: (prd_elapsed+prd_batch)].astype(np.float32)),
           'input_ids': torch.from_numpy(input_ids[prd_elapsed: (prd_elapsed+prd_batch)].astype(np.int64)),
           'input_mask': torch.from_numpy(input_mask[prd_elapsed: (prd_elapsed+prd_batch)].astype(np.int64))
               }

        targets_data = {name: tensor.cuda(device=torch.device('cuda:0'), non_blocking=True)
                               for name, tensor in targets_data.items()}
        prd_peaks.append(model(**targets_data)[0].cpu().detach().numpy())
        prd_elapsed += prd_batch
        print("Elapsed predictions: {}".format(prd_elapsed))

    prd_peaks = np.concatenate(prd_peaks)
    __, prd_peaks = cleanTapeOutput().getIntensitiesAndSpectralAngle(prd_peaks, prd_peaks, p_charges, input_ids)
    prd_peaks[prd_peaks < 0] = 0 ## Remove negative intensities
    ## Predict peaks end

    ## Create predicted spectrum
    theo_spectra = []
    for peptide, charge, prediction in zip(peptides,charges,prd_peaks):
        theo_frags = []
        for cleave in range(1, min(30,len(peptide))): ## Iterate over each fragment len 1-30
            for frag_charge in range(1,charge):
                theo_frags.append( # (m/z, intensity)
                 (mass.fast_mass(peptide[-cleave:], ion_type = 'y', charge=frag_charge, aa_mass=mod_masses),
                    prediction[(cleave-1)*6 + (frag_charge-1)])
                )
                theo_frags.append( # (m/z, intensity)
                    (mass.fast_mass(peptide[:cleave], ion_type = 'b', charge=frag_charge, aa_mass=mod_masses),
                    prediction[(cleave-1)*6 + 3 + (frag_charge-1)])
                )
        theo_frags = sorted(theo_frags)

        theo_spectra.append({ 
            'mz': np.array([ f[0] for f in theo_frags]), 
            "intensity": np.array([ f[1] for f in theo_frags]),
            "peptide" : peptide,
            "charge" : charge
            })

    return theo_spectra

def readSpectra(path, scans):
    return [] # spectra, charges, ces

def rescoreSpectra(path, peptides, scans, tolerance=0.05):
    spectra, charges, ces = readSpectra(path, scans)
    theo_spectra = predictSpectra(peptides, charges, ces)
    theo_mzs = [ ts["mz"] for ts in theo_spectra ]
    obs_matched_intensities = []
    for sought_mz, intens, mzs in zip(theo_mzs, obs_ints, obs_mzs):
        matched_spec=[]
        for m in sought_mz:
            matched_spec.append(np.max(intens[(mzs >= m - tol) & (mzs <= m + tol)], initial=0.))
        obs_matched_intensities.append(np.array(matched_spec))

    scores = []
    for theo, obs in zip(theo_intensities, obs_matched_intensities):
        theo = theo/np.sqrt(np.dot(theo,theo))
        obs = obs/np.sqrt(np.dot(obs,obs))
        cos_sim = np.dot(theo,obs)
        ang_dist = 2*np.arccos(cos_sim)/np.pi
        cross_entropy = - np.dot(theo, np.log2(obs, out=np.zeros_like(m), where=(obs!=0)))
        diff = theo-obs
        mse = np.sqrt(np.dot(diff,diff))
        imse = 1. / (1. + mse) 
        scores.append([cos_sim, ang_dist, cross_entropy, imse])

if __name__ == "__main__":
    print (predictSpectra(["IAMAPEPTIDE"],[3],[20])) 

