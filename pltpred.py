import sys
from pyteomics import mzml, mass
import numpy as np
import pandas as pd

import spectrum_utils.spectrum as sus
import spectrum_utils.plot as sup
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

def get_precursor_charge_onehot(charges, considered_charges = list(range(1,7))):
    if isinstance(charges, int):
        charges = [charges]
    array = np.zeros([len(charges), max(considered_charges)], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, precursor_charge - 1] = 1
    return array


def TokenizePeptides(peptides, tokenizer = TAPETokenizer()):
    if isinstance(peptides, str):
        peptides = [peptides]
    input_ids = pad_sequences([tokenizer.encode(p[:30]) for p in peptides])
    return input_ids, np.ones_like(input_ids)


def predictMsmsSpectrum(peptide, precursor_charge, precursor_mz, ce):
    p_charges = get_precursor_charge_onehot([precursor_charge])
    p_ces = np.hstack([ce])
    input_ids, input_mask = TokenizePeptides([peptide])
    print(input_ids, input_mask)
    model = getPrositTransformerModel()
    targets_data = { ## Load a batch of PSM CEs, charges, and peptide identifiers
        'collision_energy' : torch.FloatTensor(p_ces.astype(np.float32)),
        'charge': torch.FloatTensor(p_charges.astype(np.float32)),
        'input_ids': torch.from_numpy(input_ids.astype(np.int64)),
        'input_mask': torch.from_numpy(input_mask.astype(np.int64))
        }
    targets_data = {name: tensor.cuda(device=torch.device('cuda:0'), non_blocking=True)
                               for name, tensor in targets_data.items()}
    prd_peaks = [model(**targets_data)[0].cpu().detach().numpy()]
    prd_peaks = np.concatenate(prd_peaks)
    __, prd_peaks = cleanTapeOutput().getIntensitiesAndSpectralAngle(prd_peaks, prd_peaks, p_charges, input_ids, start_stop_token=True)
    prd_peaks[prd_peaks < 0] = 0 ## Remove negative intensities
    theo_spectra = []
    prediction = prd_peaks[0]
    theo_frags = []
    for cleave in range(1, min(30,len(peptide)-1)): ## Iterate over each fragment len 1-30
        for frag_charge in range(1,precursor_charge+1): # i.e. max fragment charge = precursor charge 
            theo_frags.append( # (m/z, intensity)
                (mass.fast_mass(peptide[-cleave:], ion_type = 'y', charge=frag_charge, aa_mass=mod_masses),
                    prediction[(cleave-1)*6 + (frag_charge-1)]))
            theo_frags.append( # (m/z, intensity)
                    (mass.fast_mass(peptide[:cleave], ion_type = 'b', charge=frag_charge, aa_mass=mod_masses),
                    prediction[(cleave-1)*6 + 3 + (frag_charge-1)]))
        theo_frags = sorted(theo_frags)
    
    spectrum = sus.MsmsSpectrum("Predicted", precursor_mz, precursor_charge,
                            [ f[0] for f in theo_frags], 
                            [ f[1] for f in theo_frags],
                            peptide=peptide).annotate_peptide_fragments(0.5, 'Da', ion_types='by',max_ion_charge=precursor_charge)

    return spectrum


def readPout(path, scan, fdr = 0.05):
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
            if int(words[fields["file_idx"]]) != 0: # Remove PSMs not from first file
                continue
            proteins = "\t".join(words[num_col-1])
            words = words[:num_col-1] + [proteins]
            content.append(words)
    # content = content[:10] # just take the first entries when debugging
    scans = [int(words[fields["scan"]]) for words in content]
    peptides = [words[fields["sequence"]] for words in content]
    charges = [int(words[fields["charge"]]) for words in content] 
    ix = scans.index(scan)    
    return scan, peptides[ix], charges[ix]       

def readSpectrum(mzml_file, scan, peptide):
    obs_ints, obs_mzs, ces = [], [], []
    with mzml.read(mzml_file, use_index=True) as spectra:
        match = spectra[scan-1] # indedxed to scan-1
        precursor = match["precursorList"]['precursor'][0]
        ion = precursor['selectedIonList']['selectedIon'][0]
        p_mz = float(ion['selected ion m/z']) # Measured mass to charge ratio of precursor ion
        p_z = int(ion['charge state']) # Precursor charge
        p_m = (p_mz - mass.Composition({'H+': 1}).mass(charege=1))*p_z # Mass of precursor
        try:
            ce = float(precursor['activation']['collision energy'])/100.
        except KeyError:
            ce=0.3    
        obs_spectrum = {
                "intensity": match['intensity array'],
                "mz" : match['m/z array'],
                "pcharge" : p_z,
                "pmz" : p_mz,
                "pmass" : p_m,
                "CE" : ce,
            }
        print(obs_spectrum)
        obsMsmsSpectrum = sus.MsmsSpectrum(peptide, p_mz, p_z, match['m/z array'], match['intensity array'], peptide=peptide).remove_precursor_peak(20., 'ppm').filter_intensity(min_intensity=0.05, max_num_peaks=50).annotate_peptide_fragments(20., 'ppm', ion_types='aby',max_ion_charge=p_z)

    return obsMsmsSpectrum, obs_spectrum

def plotMirror(spectrum_top, spectrum_bottom):
    fig, ax = plt.subplots(figsize=(12, 6))
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()

def plotSpectrum(pout_file, mzml_file, scan):
    scan, peptide, charge = readPout(pout_file, scan)
    obsMsmsSpectrum, obs_spectrum = readSpectrum(mzml_file, scan, peptide)
    predictedSpectrum = predictMsmsSpectrum(peptide, obs_spectrum["pcharge"], obs_spectrum["pmz"], obs_spectrum["CE"])
    plotMirror(obsMsmsSpectrum, predictedSpectrum)

if __name__ == "__main__":
#    plotSpectrum(
#        "data/data_yeast_casanovo/percolator.target.peptides.txt",
#        "data/data_yeast_casanovo/preproc.high.yeast.PXD003868.mzML",
#        int(69272))
    plotSpectrum("/hd2/lukask/ms/graphtest/PXD028735/crux-output/percolator.target.peptides.txt",
        "/hd2/lukask/ms/graphtest/PXD028735/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML",
        int(79530))
