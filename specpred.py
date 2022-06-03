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

def tokenizePeptides(peptides, tokenizer = TAPETokenizer()):
    if isinstance(peptides, str):
        peptides = [peptides]
    input_ids =  pad_sequences([tokenizer.encode(p[:30]) for p in peptides])
    mask = np.ones_like(input_ids)
    mask[input_ids==0] = 0
    #mask[input_ids==3] = 0
    #mask = mask[:,1:-1]
    return input_ids, mask

def predictSpectra(peptides, charges, ces = None, prediction_batch_size = 200):
    if not ces:
        ces = [0.30]*len(peptides)
    p_charges = get_precursor_charge_onehot(charges)
    p_ces = np.hstack(ces)
    input_ids, input_mask = tokenizePeptides(peptides)
    model = getPrositTransformerModel()
    prd_peaks = []
    prd_elapsed = 0
    
    while prd_elapsed < len(peptides): ## Predict until all valid PSMs are accounted for
        if (prd_elapsed + prediction_batch_size) > len(peptides): ## Truncate last prediction to match #PSMS
            prediction_batch_size = len(peptides) - prd_elapsed

        targets_data = { ## Load a batch of PSM CEs, charges, and peptide identifiers
           'collision_energy' : torch.FloatTensor(p_ces[prd_elapsed: (prd_elapsed+prediction_batch_size)].astype(np.float32)),
           'charge': torch.FloatTensor(p_charges[prd_elapsed: (prd_elapsed+prediction_batch_size)].astype(np.float32)),
           'input_ids': torch.from_numpy(input_ids[prd_elapsed: (prd_elapsed+prediction_batch_size)].astype(np.int64)),
           'input_mask': torch.from_numpy(input_mask[prd_elapsed: (prd_elapsed+prediction_batch_size)].astype(np.int64))
               }
        targets_data = {name: tensor.cuda(device=torch.device('cuda:0'), non_blocking=True)
                               for name, tensor in targets_data.items()}
        print(targets_data)
        prediction = model(**targets_data)[0].cpu().detach().numpy()
        prd_peaks.append(prediction)
        # print(prediction)
        prd_elapsed += prediction_batch_size
        print("Elapsed predictions: {}".format(prd_elapsed))

    prd_peaks = np.concatenate(prd_peaks)
    __, prd_peaks = cleanTapeOutput().getIntensitiesAndSpectralAngle(prd_peaks, prd_peaks, p_charges, input_ids, start_stop_token=True)
    prd_peaks[prd_peaks < 0] = 0 ## Remove negative intensities
    ## Predict peaks end
    print(prd_peaks)

    ## Create predicted spectrum
    theo_spectra = []
    for peptide, charge, prediction in zip(peptides,charges,prd_peaks):
        prediction = prediction.reshape((29,6))
        # print(prediction, prediction.shape, charge)
        theo_frags = []
        for cleave in range(1, min(30,len(peptide))): ## Iterate over each fragment len 1-30
            for frag_charge in range(1,min(charge+1,3)): # i.e. max fragment charge = precursor charge -1
                theo_frags.append( # (m/z, intensity)
                 (mass.fast_mass(peptide[-cleave:], ion_type = 'y', charge=frag_charge, aa_mass=mod_masses),
                    prediction[cleave-1, frag_charge-1])
                )
                theo_frags.append( # (m/z, intensity)
                    (mass.fast_mass(peptide[:cleave], ion_type = 'b', charge=frag_charge, aa_mass=mod_masses),
                    prediction[(cleave-1), 3 + (frag_charge-1)])
                )
        theo_frags = sorted(theo_frags)

        theo_spectra.append({ 
            'mz': np.array([ f[0] for f in theo_frags]), 
            "intensity": np.array([ f[1] for f in theo_frags]),
            "peptide" : peptide,
            "charge" : charge
            })

    return theo_spectra

def readPout(path, fdr = 0.05):
    with open(path) as pout:
        content = []       
        words = pout.readline().strip().split('\t')
        fields = { w : ix for ix, w in enumerate(words) }
        num_col = len(words)
        qix = fields["percolator q-value"]
        for line in pout:
            words = line.split('\t')
            if float(words[qix])>fdr:
                break
            if len(words[fields["sequence"]]) > 30: # Remove peptides longer than the net can handle
                continue
            proteins = "\t".join(words[num_col-1])
            words = words[:num_col-1] + [proteins]
            content.append(words)
    #content = content[1:2] # just take the first entries when debugging
    scans = [int(words[fields["scan"]]) for words in content]
    peptides = [words[fields["sequence"]] for words in content]
    charges = [int(words[fields["charge"]]) for words in content]     
    return scans, peptides, charges       

def readSpectra(mzml_file, scans):
    obs_spectra = []
    obs_ints, obs_mzs, ces = [], [], []
    with mzml.read(mzml_file, use_index=True) as spectra:
        for id, scan in enumerate(scans): ## Get the spectra that match the valid PSMs be scanNr
            match = spectra[scan-1] # indedxed to scan-1
            precursor = match["precursorList"]['precursor'][0]
            ion = precursor['selectedIonList']['selectedIon'][0]
            p_mz = float(ion['selected ion m/z']) # Measured mass to charge ratio of precursor ion
            p_z = int(ion['charge state']) # Precursor charge
            p_m = (p_mz - mass.Composition({'H+': 1}).mass(charege=1))*p_z # Mass of precursor
            try:
                ce = float(precursor['activation']['collision energy'])
            except KeyError:
                ce=0.3 
            obs_spectra.append({
                "intensity": match['intensity array'],
                "mz" : match['m/z array'],
                "pcharge" : p_z,
                "pmz" : p_mz,
                "pmass" : p_m,
                "CE" : ce,
            })
    return obs_spectra

def rescoreSpectra(path, peptides, scans, tolerance=0.005):
    obs_spectra = readSpectra(path, scans)
    obs_ints, obs_mzs = [ os["intensity"] for os in obs_spectra ], [ os["mz"] for os in obs_spectra ]
    charges, ces =  [ os["pcharge"] for os in obs_spectra ], [ os["CE"] for os in obs_spectra ]
    theo_spectra = predictSpectra(peptides, charges, ces)
    theo_mzs = [ ts["mz"] for ts in theo_spectra ]
    theo_intensities = [ ts["intensity"] for ts in theo_spectra ]
    obs_matched_intensities = []
    for sought_mz, intens, mzs in zip(theo_mzs, obs_ints, obs_mzs):
        matched_spec=[]
        # print(np.max(intens))
        for m in sought_mz:
            #print(m, mzs[(mzs >= m - tolerance) & (mzs <= m + tolerance)], intens[(mzs >= m - tolerance) & (mzs <= m + tolerance)])
            matched_spec.append(np.max(intens[(mzs >= m - tolerance) & (mzs <= m + tolerance)], initial=0.))
        obs_matched_intensities.append(np.array(matched_spec))

    scores = []
    for theo, obs, peptide in zip(theo_intensities, obs_matched_intensities, peptides):
        print(peptide)
        # print(theo, obs)
        theo = theo/np.sqrt(np.dot(theo,theo))
        obs = obs/np.sqrt(np.dot(obs,obs))
        #print(theo)
        #print(obs)
        cos_sim = np.dot(theo,obs)
        ang_dist = 2*np.arccos(cos_sim)/np.pi
        cross_entropy = - np.dot(theo, np.log2(obs, where=(obs!=0.))) # , out=np.zeros_like(m)
        diff = theo-obs
        mse = np.sqrt(np.dot(diff,diff))
        imse = 1. / (1. + mse)
        print(cos_sim, ang_dist, cross_entropy, mse, imse) 
        scores.append([cos_sim, ang_dist, cross_entropy, imse])
    return scores

def rescorePout(pin_path, mzml_path, fdr = 0.05):
    scans, peptides, charges = readPout(pin_path, fdr)
    return rescoreSpectra(mzml_path, peptides, scans)


if __name__ == "__main__":
#    print (predictSpectra(["IAMAPEPTIDETHATISWERYSTRETCHEDANDTHATISAPARANTLYFANTASTICLYWEIRD"],[3],[0.3])) 
    predictSpectra(["INIDHKFHRHL"],[3],np.array([0.317]))
#    print (predictSpectra(["INIDHKFHRHL"],[3],np.array([0.317])))
#    print(rescorePout(
#        "data/data_yeast_casanovo/percolator.target.peptides.txt",
#        "data/data_yeast_casanovo/preproc.high.yeast.PXD003868.mzML"))
#    rescorePout(
#        "data/data_yeast_casanovo/percolator.target.peptides.txt",
#        "data/data_yeast_casanovo/preproc.high.yeast.PXD003868.mzML")
