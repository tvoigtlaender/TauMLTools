import dask
import numpy as np
import awkward as ak
import uproot
import yaml

# np.seterr("raise")

# path_DY = "root://eoscms.cern.ch//eos/cms/store/group/phys_tau/TauML/prod_2018_v2/full_tuples/DYJetsToLL_M-50-amcatnloFXFX_ext2/eventTuple_1.root"
path_SuM = "root://eosuser.cern.ch///eos/cms/store/group/phys_tau/TauML/prod_2018_v2/ShuffleMergeSpectral_trainingSamples-2_rerun/ShuffleMergeSpectral_0.root"

cfg_path = "/work/tvoigtlaender/TauTransformerComparison/dataloader_dev/TauMLTools/cfg.yaml"

with open(cfg_path, 'r') as stream:
    feature_names = yaml.safe_load(stream)['feature_names']

use_dask = True

# if use_dask:
a = uproot.dask(f"{path_SuM}:taus",step_size="1 GB", timeout=3000)
    
# else:
    
#     a = uproot.open(f"{path_SuM}:taus",step_size="1 GB", timeout=3000)

#     used_inputs = ["genLepton_kind",'tau_gj_angle_diff', 'pfCand_pt', 'ele_full5x5_sigmaEtaEta', 'pfCand_dxy', 'ele_full5x5_e2x5Top', 'ele_eRight', 'ele_sigmaIetaIeta', 'tau_ip3d', 'ele_full5x5_e1x5', 'pv_y', 'ele_full5x5_eRight', 'ele_eBottom', 'ele_full5x5_r9', 'muon_numberOfValidHits', 'pfCand_track_chi2', 'ele_full5x5_sigmaIphiIphi', 'tau_flightLength_y', 'tau_eta', 'pfCand_eta', 'ele_e1x5', 'ele_ecalEnergy', 'muon_pfEcalEnergy', 'tau_chargedIsoPtSum', 'ele_hcalDepth1OverEcalBc', 'ele_eta', 'tau_sv_z', 'tau_flightLength_z', 'ele_closestCtfTrack_numberOfValidHits', 'ele_cc_gamma_energy', 'ele_full5x5_e2x5Bottom', 'tau_neutralIsoPtSumdR03', 'tau_dxy_error', 'ele_full5x5_e2x5Left', 'tau_leadChargedCand_etaAtEcalEntrance', 'ele_full5x5_hcalDepth1OverEcalBc', 'tau_decayMode', 'ele_full5x5_sigmaIetaIphi', 'pfCand_dz_error', 'pfCand_hasTrackDetails', 'pfCand_dz', 'ele_full5x5_hcalDepth2OverEcal', 'ele_ecalEnergy_error', 'ele_sigmaIphiIphi', 'ele_sigmaEtaEta', 'tau_ip3d_error', 'muon_phi', 'ele_cc_n_gamma', 'tau_flightLength_x', 'tau_pt', 'ele_e2x5Max', 'ele_trackMomentumAtCalo', 'ele_full5x5_hcalDepth1OverEcal', 'tau_neutralIsoPtSumWeight', 'ele_trackMomentumAtVtxWithConstraint', 'muon_pt', 'ele_trackMomentumAtVtx', 'ele_full5x5_e2x5Right', 'pfCand_vertex_z', 'pv_x', 'ele_trackMomentumOut', 'tau_dz', 'ele_r9', 'tau_neutralIsoPtSum', 'ele_full5x5_eBottom', 'ele_full5x5_eLeft', 'ele_eLeft', 'muon_dxy', 'ele_hcalDepth1OverEcal', 'muon_dxy_error', 'pfCand_vertex_y', 'muon_eta', 'ele_full5x5_eTop', 'pfCand_track_ndof', 'ele_eTop', 'ele_full5x5_e2x5Max', 'pv_z', 'ele_full5x5_hcalDepth2OverEcalBc', 'ele_gsfTrack_pt_error', 'muon_normalizedChi2', 'ele_trackMomentumAtEleClus', 'ele_hcalDepth2OverEcal', 'tau_e_ratio', 'pfCand_vertex_x', 'tau_sv_y', 'tau_hasSecondaryVertex', 'ele_gsfTrack_pt', 'tau_dxy', 'pfCand_dxy_error', 'ele_phi', 'ele_closestCtfTrack_normalizedChi2', 'ele_cc_ele_energy', 'tau_sv_x', 'ele_e5x5', 'pfCand_phi', 'ele_pt', 'ele_sigmaIetaIphi', 'ele_hcalDepth2OverEcalBc', 'tau_chargedIsoPtSumdR03', 'pfCand_particleType', 'ele_full5x5_sigmaIetaIeta', 'tau_neutralIsoPtSumWeightdR03', 'tau_dz_error', 'ele_full5x5_e5x5', 'tau_phi', 'tau_mass']
    
#     a = a.arrays(used_inputs, library="ak", how="zip")

def get_type(col):
    # Get the primitive type of the given column
    if col.ndim == 1:
        return col.type.content.primitive
    elif col.ndim == 2:
        return col.type.content.content.primitive
    else:
        raise ValueError(f"Unsupported dimension: {col.ndim}")

def is_finite(col, dask_array=True):
    # Get the min and max values for the given column
    primitive_type_col = get_type(col)
    if np.issubdtype(primitive_type_col, np.integer):
        min_value = np.iinfo(primitive_type_col).min
        max_value = np.iinfo(primitive_type_col).max
    elif np.issubdtype(primitive_type_col, np.floating):
        min_value = np.finfo(primitive_type_col).min
        max_value = np.finfo(primitive_type_col).max
    else:
        raise TypeError(f"Unsupported DataType: {primitive_type_col}")
    # Check if the values are infinite 
    return np.isfinite(col) & (col!=min_value) & (col!=max_value)

def has_values(col):
    # Check 
    if col.ndim == 2:
        return ak.flatten(ak.argmax(col, keepdims=True, axis=-1, mask_identity=False)) != -1
    else:
        raise ValueError(f"Unsupported dimension: {col.ndim}")

#def preprocess_array(a, feature_names, add_feature_names, verbose=False):

# Filter non_valid tau candidates with no tau information
a = a[is_finite(a["tau_pt"])]

# dictionary to store preprocessed features (per feature type)
a_preprocessed = {feature_type: {} for feature_type in feature_names.keys()}

# fill lazily original features which don't require preprocessing  
for feature_type, feature_list in feature_names.items():
    for feature_name in feature_list:
        try:
            f = f'{feature_type}_{feature_name}' if feature_type != 'global' else feature_name
            a_preprocessed[feature_type][feature_name] = a[f]
        except:
            print(f'        {f} not found in input ROOT file, will skip it') 


# a_done = dask.compute(a_preprocessed)[0]

# for p_type in a_done.keys():
#     for f_name in a_done[p_type].keys():
#         print(f"{p_type}_{f_name}", ak.all(is_finite(a_done[p_type][f_name])))

# exit(0)

a_preprocessed['global']['tau_E_over_pt'] = np.sqrt((a['tau_pt']*np.cosh(a['tau_eta']))*(a['tau_pt']*np.cosh(a['tau_eta'])) + a['tau_mass']*a['tau_mass'])/a['tau_pt']
a_preprocessed['global']['tau_n_charged_prongs'] = a['tau_decayMode']//5 + 1
a_preprocessed['global']['tau_n_neutral_prongs'] = a['tau_decayMode']%5
a_preprocessed['global']['tau_chargedIsoPtSumdR03_over_dR05'] = ak.where((a['tau_chargedIsoPtSum']!=0), (a['tau_chargedIsoPtSumdR03']/a['tau_chargedIsoPtSum']), 0)
a_preprocessed['global']['tau_neutralIsoPtSumWeight_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumWeight']/a['tau_neutralIsoPtSum']), 0)
a_preprocessed['global']['tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumWeightdR03']/a['tau_neutralIsoPtSum']), 0)
a_preprocessed['global']['tau_neutralIsoPtSumdR03_over_dR05'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumdR03']/a['tau_neutralIsoPtSum']), 0)

tau_dxy_valid = (is_finite(a['tau_dxy']) & (a['tau_dxy'] > -10) & is_finite(a['tau_dxy_error']) & (a['tau_dxy_error'] > 0)) #Used
a_preprocessed['global']['tau_dxy_valid'] = ak.values_astype(tau_dxy_valid, np.float32)
a_preprocessed['global']['tau_dxy'] = ak.where(tau_dxy_valid, (a['tau_dxy']), 0)
a_preprocessed['global']['tau_dxy_sig'] = ak.where(tau_dxy_valid, (np.abs(a['tau_dxy'])/a['tau_dxy_error']), 0)

a_preprocessed['global']['tau_ip3d_sig'] = np.abs(a['tau_ip3d'])/a['tau_ip3d_error']

tau_dz_sig_valid = (is_finite(a['tau_dz']) & is_finite(a['tau_dz_error']) & (a['tau_dz_error'] > 0)) #Used
a_preprocessed['global']['tau_dz_sig_valid'] = ak.values_astype(tau_dz_sig_valid, np.float32)
# a_preprocessed['global']['tau_dz'] = ak.where(tau_dz_sig_valid, (a['tau_dz']), 0)
a_preprocessed['global']['tau_dz_sig'] = ak.where(tau_dz_sig_valid, (np.abs(a['tau_dz'])/a['tau_dz_error']), 0)

tau_gj_angle_diff_valid = (a['tau_gj_angle_diff'] >= 0) # Used
a_preprocessed['global']['tau_gj_angle_diff_valid'] = ak.values_astype(tau_gj_angle_diff_valid, np.float32)
a_preprocessed['global']['tau_gj_angle_diff'] = ak.where(tau_gj_angle_diff_valid, (a['tau_gj_angle_diff']), 0)

a_preprocessed['global']['tau_leadChargedCand_etaAtEcalEntrance_minus_tau_eta'] = a['tau_leadChargedCand_etaAtEcalEntrance'] - a['tau_eta']
a_preprocessed['global']['particle_type'] = 9*ak.ones_like(a['tau_pt']) # assign unique particle type to a "global" token 

# ------- PF candidates ------- #

# shift delta phi into [-pi, pi] range 
pf_dphi = (a['pfCand_phi'] - a['tau_phi'])
pf_dphi = ak.where(pf_dphi <= np.pi, pf_dphi, pf_dphi - 2*np.pi)
pf_dphi = ak.where(pf_dphi >= -np.pi, pf_dphi, pf_dphi + 2*np.pi)
pf_deta = (a['pfCand_eta'] - a['tau_eta'])

a_preprocessed['pfCand']['dphi'] = pf_dphi
a_preprocessed['pfCand']['deta'] = pf_deta
a_preprocessed['pfCand']['rel_pt'] = a['pfCand_pt'] / a['tau_pt']
a_preprocessed['pfCand']['r'] = np.sqrt(np.square(pf_deta) + np.square(pf_dphi))
a_preprocessed['pfCand']['theta'] = np.arctan2(pf_dphi, pf_deta) # dphi -> y, deta -> x
a_preprocessed['pfCand']['particle_type'] = a['pfCand_particleType'] - 1

vertex_z_valid = is_finite(a['pfCand_vertex_z']) #Used
a_preprocessed['pfCand']['vertex_dx'] = a['pfCand_vertex_x'] - a['pv_x']
a_preprocessed['pfCand']['vertex_dy'] = a['pfCand_vertex_y'] - a['pv_y']
a_preprocessed['pfCand']['vertex_dz'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z']), 0)
a_preprocessed['pfCand']['vertex_dx_tauFL'] = a['pfCand_vertex_x'] - a['pv_x'] - a['tau_flightLength_x']
a_preprocessed['pfCand']['vertex_dy_tauFL'] = a['pfCand_vertex_y'] - a['pv_y'] - a['tau_flightLength_y']
a_preprocessed['pfCand']['vertex_dz_tauFL'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z'] - a['tau_flightLength_z']), 0)

has_track_details = (a['pfCand_hasTrackDetails'] == 1)
has_track_details_track_ndof = has_track_details * (a['pfCand_track_ndof'] > 0) # Used
has_track_details_dxy_sig_finite = has_track_details * (is_finite(a['pfCand_dxy']) & is_finite(a['pfCand_dxy_error'])) #Used
has_track_details_dz_sig_finite = has_track_details * (is_finite(a['pfCand_dz']) & is_finite(a['pfCand_dz_error'])) #Used
a_preprocessed['pfCand']['dxy_sig_valid'] = ak.values_astype(has_track_details_dxy_sig_finite, np.float32)
a_preprocessed['pfCand']['dxy_sig'] = ak.where(has_track_details_dxy_sig_finite, (np.abs(a['pfCand_dxy'])/a['pfCand_dxy_error']), 0)
a_preprocessed['pfCand']['dz_sig_valid'] = ak.values_astype(has_track_details_dz_sig_finite, np.float32)
a_preprocessed['pfCand']['dz_sig'] = ak.where(has_track_details_dz_sig_finite, (np.abs(a['pfCand_dz'])/a['pfCand_dz_error']), 0)
a_preprocessed['pfCand']['track_ndof_valid'] = ak.values_astype(has_track_details_track_ndof, np.float32)
a_preprocessed['pfCand']['track_ndof'] = ak.where(has_track_details_track_ndof, a['pfCand_track_ndof'], 0)
a_preprocessed['pfCand']['chi2_ndof'] = ak.where(has_track_details_track_ndof, (a['pfCand_track_chi2']/a['pfCand_track_ndof']), 0)

# ------- Electrons ------- #

# shift delta phi into [-pi, pi] range 
ele_dphi = (a['ele_phi'] - a['tau_phi'])
ele_dphi = ak.where(ele_dphi <= np.pi, ele_dphi, ele_dphi - 2*np.pi)
ele_dphi = ak.where(ele_dphi >= -np.pi, ele_dphi, ele_dphi + 2*np.pi)
ele_deta = (a['ele_eta'] - a['tau_eta'])

a_preprocessed['ele']['dphi'] = ele_dphi
a_preprocessed['ele']['deta'] = ele_deta
a_preprocessed['ele']['rel_pt'] = a['ele_pt'] / a['tau_pt']
a_preprocessed['ele']['r'] = np.sqrt(np.square(ele_deta) + np.square(ele_dphi))
a_preprocessed['ele']['theta'] = np.arctan2(ele_dphi, ele_deta) # dphi -> y, deta -> x
a_preprocessed['ele']['particle_type'] = 7*ak.ones_like(a['ele_pt']) # assuming PF candidate types are [0..6]

ele_cc_valid = (a['ele_cc_ele_energy'] >= 0) #Used
a_preprocessed['ele']['cc_valid'] = ak.values_astype(ele_cc_valid, np.float32)
a_preprocessed['ele']['cc_ele_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_ele_energy']/a['ele_pt']), 0)
a_preprocessed['ele']['cc_gamma_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_gamma_energy']/a['ele_cc_ele_energy']), 0)
a_preprocessed['ele']['cc_n_gamma'] = ak.where(ele_cc_valid, a['ele_cc_n_gamma'], 0)
a_preprocessed['ele']['rel_trackMomentumAtVtx'] = a['ele_trackMomentumAtVtx']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumAtCalo'] = a['ele_trackMomentumAtCalo']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumOut'] = a['ele_trackMomentumOut']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumAtEleClus'] = a['ele_trackMomentumAtEleClus']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumAtVtxWithConstraint'] = a['ele_trackMomentumAtVtxWithConstraint']/a['ele_pt']
a_preprocessed['ele']['rel_ecalEnergy'] = a['ele_ecalEnergy']/a['ele_pt']
a_preprocessed['ele']['ecalEnergy_sig'] = a['ele_ecalEnergy']/a['ele_ecalEnergy_error']
a_preprocessed['ele']['rel_gsfTrack_pt'] = a['ele_gsfTrack_pt']/a['ele_pt']
a_preprocessed['ele']['gsfTrack_pt_sig'] = a['ele_gsfTrack_pt']/a['ele_gsfTrack_pt_error']

ele_has_closestCtfTrack = (a['ele_closestCtfTrack_normalizedChi2'] >= 0) #Used
a_preprocessed['ele']['has_closestCtfTrack'] = ak.values_astype(ele_has_closestCtfTrack, np.float32)
a_preprocessed['ele']['closestCtfTrack_normalizedChi2'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_normalizedChi2'], 0)
a_preprocessed['ele']['closestCtfTrack_numberOfValidHits'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_numberOfValidHits'], 0)

ele_mva_valid = is_finite(a['ele_e5x5']) #Used
ele_features = ['sigmaEtaEta', 'sigmaIetaIeta', 'sigmaIphiIphi', 'sigmaIetaIphi', 'e1x5', 'e2x5Max', 'e5x5', 'r9', 'hcalDepth1OverEcal', 'hcalDepth2OverEcal', 'hcalDepth1OverEcalBc', 'hcalDepth2OverEcalBc','eLeft', 'eRight', 'eBottom', 'eTop','full5x5_sigmaEtaEta', 'full5x5_sigmaIetaIeta', 'full5x5_sigmaIphiIphi', 'full5x5_sigmaIetaIphi','full5x5_e1x5', 'full5x5_e2x5Max', 'full5x5_e5x5', 'full5x5_r9','full5x5_hcalDepth1OverEcal', 'full5x5_hcalDepth2OverEcal', 'full5x5_hcalDepth1OverEcalBc', 'full5x5_hcalDepth2OverEcalBc','full5x5_eLeft', 'full5x5_eRight', 'full5x5_eBottom', 'full5x5_eTop','full5x5_e2x5Left', 'full5x5_e2x5Right', 'full5x5_e2x5Bottom', 'full5x5_e2x5Top']
a_preprocessed['ele']['mva_valid'] = ak.values_astype(ele_mva_valid, np.float32)
for ele_feature in ele_features:
    a_preprocessed['ele'][ele_feature] = ak.where(ele_mva_valid, a[f'ele_{ele_feature}'], 0)

# ------- Muons ------- #

# shift delta phi into [-pi, pi] range 
muon_dphi = (a['muon_phi'] - a['tau_phi'])
muon_dphi = ak.where(muon_dphi <= np.pi, muon_dphi, muon_dphi - 2*np.pi)
muon_dphi = ak.where(muon_dphi >= -np.pi, muon_dphi, muon_dphi + 2*np.pi)
muon_deta = (a['muon_eta'] - a['tau_eta'])

a_preprocessed['muon']['dphi'] = muon_dphi
a_preprocessed['muon']['deta'] = muon_deta
a_preprocessed['muon']['rel_pt'] = a['muon_pt'] / a['tau_pt']
a_preprocessed['muon']['r'] = np.sqrt(np.square(muon_deta) + np.square(muon_dphi))
a_preprocessed['muon']['theta'] = np.arctan2(muon_dphi, muon_deta) # dphi -> y, deta -> x
a_preprocessed['muon']['particle_type'] = 8*ak.ones_like(a['muon_pt']) # assuming PF candidate types are [0..6]

a_preprocessed['muon']['dxy_sig'] = np.abs(a['muon_dxy'])/a['muon_dxy_error']

muon_normalizedChi2_valid = ((a['muon_normalizedChi2'] > 0) * is_finite(a['muon_normalizedChi2'])) #Used
a_preprocessed['muon']['normalizedChi2_valid'] = ak.values_astype(muon_normalizedChi2_valid, np.float32)
a_preprocessed['muon']['normalizedChi2'] = ak.where(muon_normalizedChi2_valid, a['muon_normalizedChi2'], 0)
a_preprocessed['muon']['numberOfValidHits'] = ak.where(muon_normalizedChi2_valid, a['muon_numberOfValidHits'], 0)

a_preprocessed['muon']['rel_pfEcalEnergy'] = a['muon_pfEcalEnergy']/a['muon_pt']

# add_columns = {_f: a[_f] for _f in add_feature_names} if add_feature_names is not None else None
# if use_dask:
a_done = dask.compute(a_preprocessed)[0]
# else:
#     a_done = a_preprocessed

for p_type in a_done.keys():
    for f_name in a_done[p_type].keys():
        print(f"{p_type}_{f_name}", ak.all(is_finite(a_done[p_type][f_name])))
# print(a_done)

print("Done")