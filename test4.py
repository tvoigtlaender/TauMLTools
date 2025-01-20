import dask
import numpy as np
import awkward as ak
import uproot

# np.seterr("raise")

path_DY = "root://eoscms.cern.ch//eos/cms/store/group/phys_tau/TauML/prod_2018_v2/full_tuples/DYJetsToLL_M-50-amcatnloFXFX_ext2/eventTuple_1.root"
path_SuM = "root://eosuser.cern.ch///eos/cms/store/group/phys_tau/TauML/prod_2018_v2/ShuffleMergeSpectral_trainingSamples-2_rerun/ShuffleMergeSpectral_0.root"

a = uproot.dask(f"{path_SuM}:taus",step_size="1 GB", timeout=300)

used_inputs = ["genLepton_kind",'tau_gj_angle_diff', 'pfCand_pt', 'ele_full5x5_sigmaEtaEta', 'pfCand_dxy', 'ele_full5x5_e2x5Top', 'ele_eRight', 'ele_sigmaIetaIeta', 'tau_ip3d', 'ele_full5x5_e1x5', 'pv_y', 'ele_full5x5_eRight', 'ele_eBottom', 'ele_full5x5_r9', 'muon_numberOfValidHits', 'pfCand_track_chi2', 'ele_full5x5_sigmaIphiIphi', 'tau_flightLength_y', 'tau_eta', 'pfCand_eta', 'ele_e1x5', 'ele_ecalEnergy', 'muon_pfEcalEnergy', 'tau_chargedIsoPtSum', 'ele_hcalDepth1OverEcalBc', 'ele_eta', 'tau_sv_z', 'tau_flightLength_z', 'ele_closestCtfTrack_numberOfValidHits', 'ele_cc_gamma_energy', 'ele_full5x5_e2x5Bottom', 'tau_neutralIsoPtSumdR03', 'tau_dxy_error', 'ele_full5x5_e2x5Left', 'tau_leadChargedCand_etaAtEcalEntrance', 'ele_full5x5_hcalDepth1OverEcalBc', 'tau_decayMode', 'ele_full5x5_sigmaIetaIphi', 'pfCand_dz_error', 'pfCand_hasTrackDetails', 'pfCand_dz', 'ele_full5x5_hcalDepth2OverEcal', 'ele_ecalEnergy_error', 'ele_sigmaIphiIphi', 'ele_sigmaEtaEta', 'tau_ip3d_error', 'muon_phi', 'ele_cc_n_gamma', 'tau_flightLength_x', 'tau_pt', 'ele_e2x5Max', 'ele_trackMomentumAtCalo', 'ele_full5x5_hcalDepth1OverEcal', 'tau_neutralIsoPtSumWeight', 'ele_trackMomentumAtVtxWithConstraint', 'muon_pt', 'ele_trackMomentumAtVtx', 'ele_full5x5_e2x5Right', 'pfCand_vertex_z', 'pv_x', 'ele_trackMomentumOut', 'tau_dz', 'ele_r9', 'tau_neutralIsoPtSum', 'ele_full5x5_eBottom', 'ele_full5x5_eLeft', 'ele_eLeft', 'muon_dxy', 'ele_hcalDepth1OverEcal', 'muon_dxy_error', 'pfCand_vertex_y', 'muon_eta', 'ele_full5x5_eTop', 'pfCand_track_ndof', 'ele_eTop', 'ele_full5x5_e2x5Max', 'pv_z', 'ele_full5x5_hcalDepth2OverEcalBc', 'ele_gsfTrack_pt_error', 'muon_normalizedChi2', 'ele_trackMomentumAtEleClus', 'ele_hcalDepth2OverEcal', 'tau_e_ratio', 'pfCand_vertex_x', 'tau_sv_y', 'tau_hasSecondaryVertex', 'ele_gsfTrack_pt', 'tau_dxy', 'pfCand_dxy_error', 'ele_phi', 'ele_closestCtfTrack_normalizedChi2', 'ele_cc_ele_energy', 'tau_sv_x', 'ele_e5x5', 'pfCand_phi', 'ele_pt', 'ele_sigmaIetaIphi', 'ele_hcalDepth2OverEcalBc', 'tau_chargedIsoPtSumdR03', 'pfCand_particleType', 'ele_full5x5_sigmaIetaIeta', 'tau_neutralIsoPtSumWeightdR03', 'tau_dz_error', 'ele_full5x5_e5x5', 'tau_phi', 'tau_mass']

['dxy', 'tau_mass', 'tau_chargedIsoPtSumdR03_over_dR05', 'tau_pt_weighted_dphi_strip', 'tau_neutralIsoPtSumdR03_over_dR05', 'full5x5_hcalDepth2OverEcalBc', 'eSeedClusterOverP', 'mvaInput_sigmaEtaEta', 'vertex_dx', 'tau_n_photons', 'r', 'lostInnerHits', 'particle_type', 'cc_n_gamma', 'eLeft', 'full5x5_sigmaIphiIphi', 'cc_gamma_rel_energy', 'vertex_dz', 'cc_ele_rel_energy', 'tau_pt_weighted_deta_strip', 'hcalDepth2OverEcalBc', 'n_matches_CSC_2', 'full5x5_eLeft', 'full5x5_eRight', 'tau_leadingTrackNormChi2', 'tau_leadChargedCand_etaAtEcalEntrance_minus_tau_eta', 'full5x5_e2x5Left', 'tau_n_charged_prongs', 'n_matches_RPC_4', 'sigmaIetaIphi', 'full5x5_hcalDepth1OverEcal', 'tau_gj_angle_diff_valid', 'tau_E_over_pt', 'tau_pt', 'track_ndof', 'rel_gsfTrack_pt', 'tau_dz_sig', 'full5x5_e2x5Bottom', 'n_hits_CSC_1', 'rel_trackMomentumOut', 'tau_ip3d_valid', 'full5x5_hcalDepth2OverEcal', 'n_hits_DT_4', 'tau_charge', 'hcalDepth1OverEcalBc', 'tau_chargedIsoPtSum', 'tau_ip3d', 'tau_dz', 'full5x5_e2x5Max', 'n_hits_DT_2', 'full5x5_e5x5', 'n_hits_RPC_4', 'n_hits_RPC_2', 'rel_pt', 'tau_dz_sig_valid', 'hasTrackDetails', 'chi2_ndof', 'full5x5_eBottom', 'theta', 'n_matches_DT_3', 'tau_dxy', 'tau_dxy_sig', 'pvAssociationQuality', 'tauLeadChargedHadrCand', 'e1x5', 'vertex_dy', 'vertex_dx_tauFL', 'e5x5', 'eRight', 'n_hits_RPC_3', 'tau_e_ratio_valid', 'mvaInput_earlyBrem', 'nPixelHits', 'sigmaEtaEta', 'rel_pfEcalEnergy', 'n_matches_DT_1', 'vertex_dy_tauFL', 'cc_valid', 'deltaPhiEleClusterTrackAtCalo', 'tau_dxy_valid', 'tau_gj_angle_diff', 'tau_n_neutral_prongs', 'gsfTrack_normalizedChi2', 'tau_pt_weighted_dr_iso', 'n_matches_RPC_2', 'tau_flightLength_x', 'n_matches_CSC_3', 'charge', 'tau_flightLength_z', 'n_matches_CSC_1', 'deltaEtaSeedClusterTrackAtCalo', 'tau_eta', 'eEleClusterOverPout', 'eTop', 'tau_flightLength_sig', 'tau_ip3d_sig', 'puppiWeight', 'sigmaIetaIeta', 'pfEcalEnergy_valid', 'full5x5_e1x5', 'tau_photonPtSumOutsideSignalCone', 'deltaPhiSuperClusterTrackAtVtx', 'mvaInput_hadEnergy', 'tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum', 'tau_emFraction', 'full5x5_sigmaEtaEta', 'full5x5_e2x5Right', 'normalizedChi2_valid', 'tau_neutralIsoPtSumWeight_over_neutralIsoPtSum', 'full5x5_e2x5Top', 'e2x5Max', 'caloCompatibility', 'n_hits_CSC_3', 'dz', 'closestCtfTrack_normalizedChi2', 'tau_sv_minus_pv_y', 'eSuperClusterOverP', 'tau_e_ratio', 'full5x5_r9', 'tau_flightLength_y', 'hcalFraction', 'gsfTrack_pt_sig', 'rel_trackMomentumAtEleClus', 'nHits', 'r9', 'mvaInput_lateBrem', 'has_closestCtfTrack', 'tau_sv_minus_pv_x', 'segmentCompatibility', 'tau_hasSecondaryVertex', 'rel_trackMomentumAtVtx', 'rawCaloFraction', 'nPixelLayers', 'numberOfValidHits', 'deltaEtaEleClusterTrackAtCalo', 'nStripLayers', 'n_matches_RPC_3', 'normalizedChi2', 'n_matches_DT_4', 'full5x5_sigmaIetaIeta', 'eSeedClusterOverPout', 'vertex_dz_tauFL', 'n_hits_CSC_2', 'deltaEtaSuperClusterTrackAtVtx', 'eBottom', 'n_hits_CSC_4', 'full5x5_sigmaIetaIphi', 'n_hits_DT_3', 'tau_sv_minus_pv_z', 'n_hits_RPC_1', 'sigmaIphiIphi', 'rel_trackMomentumAtVtxWithConstraint', 'n_matches_DT_2', 'tau_neutralIsoPtSum', 'full5x5_hcalDepth1OverEcalBc', 'hcalDepth2OverEcal', 'full5x5_eTop', 'fromPV', 'closestCtfTrack_numberOfValidHits', 'n_matches_CSC_4', 'tau_footprintCorrection', 'gsfTrack_numberOfValidHits', 'tau_pt_weighted_dr_signal', 'tau_inside_ecal_crack', 'tau_puCorrPtSum', 'mvaInput_deltaEta', 'ecalEnergy_sig', 'n_hits_DT_1', 'rawHcalFraction', 'rho', 'deltaPhiSeedClusterTrackAtCalo', 'rel_trackMomentumAtCalo', 'rel_ecalEnergy', 'n_matches_RPC_1', 'dz_sig', 'hcalDepth1OverEcal', 'dxy_sig']

def is_ak_list(x):
    # Check if x is an (dask) awkward array of lists
    return isinstance(x.type.content, ak.types.listtype.ListType)

def get_type(x):
    # Get the primitive type of the given column
    if is_ak_list(x):
        return x.type.content.content.primitive
    else:
        return x.type.content.primitive

def is_finite(x):
    # Get the min and max values for the given column
    primitive_type_x = get_type(x)
    if np.issubdtype(primitive_type_x, np.integer):
        min_value = np.iinfo(primitive_type_x).min
        max_value = np.iinfo(primitive_type_x).max
    elif np.issubdtype(primitive_type_x, np.floating):
        min_value = np.finfo(primitive_type_x).min
        max_value = np.finfo(primitive_type_x).max
    else:
        raise TypeError(f"Unsupported NumpyType: {primitive_type_x}")
    # Check if the values are infinite 
    return np.isfinite(x) & (x!=min_value) & (x!=max_value)

def near_inf_to_nan(x):
    return ak.where(is_finite(x), x, np.nan)
        

#def preprocess_array(a, feature_names, add_feature_names, verbose=False):

# Filter non_valid tau candidates with no tau information
a = a[is_finite(a["tau_pt"])]

# # Filter for min_value of float32
# b = {}
# for input_i in used_inputs:
#     if (input_i.startswith("pfCand_") or input_i.startswith("ele_") or input_i.startswith("muon_")):
#         pass
#         #b[input_i]=c[input_i][ak.all(c[input_i]!=-3.4028235e+38,axis=1)]
#     else:
#         # pass
#         b[input_i]=c[input_i][is_finite(c[input_i])]
# b_done = dask.compute(b)
# b_valid = [(i,len(j)) for i,j in b_done[0].items()]
# print(b_valid)

# dictionary to store preprocessed features (per feature type)
a_preprocessed = {feature_type: {} for feature_type in feature_names.keys()}
    
# fill lazily original features which don't require preprocessing  
for feature_type, feature_list in feature_names.items():
    for feature_name in feature_list:
        try:
            f = f'{feature_type}_{feature_name}' if feature_type != 'global' else feature_name
            a_preprocessed[feature_type][feature_name] = a[f]
        except:
            if verbose:
                print(f'        {f} not found in input ROOT file, will skip it') 

a_preprocessed['global']['tau_E_over_pt'] = np.sqrt((a['tau_pt']*np.cosh(a['tau_eta']))*(a['tau_pt']*np.cosh(a['tau_eta'])) + a['tau_mass']*a['tau_mass'])/a['tau_pt']
a_preprocessed['global']['tau_n_charged_prongs'] = a['tau_decayMode']//5 + 1
a_preprocessed['global']['tau_n_neutral_prongs'] = a['tau_decayMode']%5
a_preprocessed['global']['tau_chargedIsoPtSumdR03_over_dR05'] = ak.where((a['tau_chargedIsoPtSum']!=0), (a['tau_chargedIsoPtSumdR03']/a['tau_chargedIsoPtSum']), 0)
a_preprocessed['global']['tau_neutralIsoPtSumWeight_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumWeight']/a['tau_neutralIsoPtSum']), 0)
a_preprocessed['global']['tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumWeightdR03']/a['tau_neutralIsoPtSum']), 0)
a_preprocessed['global']['tau_neutralIsoPtSumdR03_over_dR05'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumdR03']/a['tau_neutralIsoPtSum']), 0)

a_preprocessed['global']['tau_sv_minus_pv_x'] = ak.where((a['tau_hasSecondaryVertex']), (a['tau_sv_x']-a['pv_x']), 0)
a_preprocessed['global']['tau_sv_minus_pv_y'] = ak.where((a['tau_hasSecondaryVertex']), (a['tau_sv_y']-a['pv_y']), 0)
a_preprocessed['global']['tau_sv_minus_pv_z'] = ak.where((a['tau_hasSecondaryVertex']), (a['tau_sv_z']-a['pv_z']), 0)

tau_dxy_valid = (is_finite(a['tau_dxy']) & (a['tau_dxy'] > -10) & is_finite(a['tau_dxy_error']) & (a['tau_dxy_error'] > 0))
a_preprocessed['global']['tau_dxy_valid'] = ak.values_astype(tau_dxy_valid, np.float32)
a_preprocessed['global']['tau_dxy'] = ak.where(tau_dxy_valid, (a['tau_dxy']), 0)
a_preprocessed['global']['tau_dxy_sig'] = ak.where(tau_dxy_valid, (np.abs(a['tau_dxy'])/a['tau_dxy_error']), 0)

tau_ip3d_valid = (is_finite(a['tau_ip3d']) & (a['tau_ip3d'] > -10) & is_finite(a['tau_ip3d_error']) & (a['tau_ip3d_error'] > 0))
a_preprocessed['global']['tau_ip3d_valid'] = ak.values_astype(tau_ip3d_valid, np.float32)
a_preprocessed['global']['tau_ip3d'] = ak.where(tau_ip3d_valid, (a['tau_ip3d']), 0)
a_preprocessed['global']['tau_ip3d_sig'] = ak.where(tau_ip3d_valid, (np.abs(a['tau_ip3d'])/a['tau_ip3d_error']), 0)

tau_dz_sig_valid = (is_finite(a['tau_dz']) & is_finite(a['tau_dz_error']) & (a['tau_dz_error'] > 0))
a_preprocessed['global']['tau_dz_sig_valid'] = ak.values_astype(tau_dz_sig_valid, np.float32)
# a_preprocessed['global']['tau_dz'] = ak.where(tau_dz_sig_valid, (a['tau_dz']), 0)
a_preprocessed['global']['tau_dz_sig'] = ak.where(tau_dz_sig_valid, (np.abs(a['tau_dz'])/a['tau_dz_error']), 0)

tau_e_ratio_valid = (is_finite(a['tau_e_ratio']) & ['tau_e_ratio'] > 0)
a_preprocessed['global']['tau_e_ratio_valid'] = ak.values_astype(tau_e_ratio_valid, np.float32)
a_preprocessed['global']['tau_e_ratio'] = ak.where(tau_e_ratio_valid, (a['tau_e_ratio']), 0)

tau_gj_angle_diff_valid = ((is_finite(a['tau_gj_angle_diff']) | a['tau_gj_angle_diff']==0) & a['tau_gj_angle_diff'] >= 0)
a_preprocessed['global']['tau_gj_angle_diff_valid'] = ak.values_astype(tau_gj_angle_diff_valid, np.float32)
a_preprocessed['global']['tau_gj_angle_diff'] = ak.where(tau_gj_angle_diff_valid, (a['tau_gj_angle_diff']), -1)

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

vertex_z_valid = is_finite( a['pfCand_vertex_z'])
a_preprocessed['pfCand']['vertex_dx'] = a['pfCand_vertex_x'] - a['pv_x']
a_preprocessed['pfCand']['vertex_dy'] = a['pfCand_vertex_y'] - a['pv_y']
a_preprocessed['pfCand']['vertex_dz'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z']), -10)
a_preprocessed['pfCand']['vertex_dx_tauFL'] = a['pfCand_vertex_x'] - a['pv_x'] - a['tau_flightLength_x']
a_preprocessed['pfCand']['vertex_dy_tauFL'] = a['pfCand_vertex_y'] - a['pv_y'] - a['tau_flightLength_y']
a_preprocessed['pfCand']['vertex_dz_tauFL'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z'] - a['tau_flightLength_z']), -10)

has_track_details = (a['pfCand_hasTrackDetails'] == 1)
has_track_details_track_ndof = has_track_details * (a['pfCand_track_ndof'] > 0)
has_track_details_dxy_finite = has_track_details * is_finite(a['pfCand_dxy'])
has_track_details_dxy_sig_finite = has_track_details * is_finite(np.abs(a['pfCand_dxy'])/a['pfCand_dxy_error'])
has_track_details_dz_finite = has_track_details * is_finite(a['pfCand_dz'])
has_track_details_dz_sig_finite = has_track_details * is_finite(np.abs(a['pfCand_dz'])/a['pfCand_dz_error'])
a_preprocessed['pfCand']['dxy'] = ak.where(has_track_details_dxy_finite, a['pfCand_dxy'], 0)
a_preprocessed['pfCand']['dxy_sig'] = ak.where(has_track_details_dxy_sig_finite, (np.abs(a['pfCand_dxy'])/a['pfCand_dxy_error']), 0)
a_preprocessed['pfCand']['dz'] = ak.where(has_track_details_dz_finite, a['pfCand_dz'], 0)
a_preprocessed['pfCand']['dz_sig'] = ak.where(has_track_details_dz_sig_finite, (np.abs(a['pfCand_dz'])/a['pfCand_dz_error']), 0)
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

ele_cc_valid = (a['ele_cc_ele_energy'] >= 0)
a_preprocessed['ele']['cc_valid'] = ak.values_astype(ele_cc_valid, np.float32)
a_preprocessed['ele']['cc_ele_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_ele_energy']/a['ele_pt']), 0)
a_preprocessed['ele']['cc_gamma_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_gamma_energy']/a['ele_cc_ele_energy']), 0)
a_preprocessed['ele']['cc_n_gamma'] = ak.where(ele_cc_valid, a['ele_cc_n_gamma'], -1)
a_preprocessed['ele']['rel_trackMomentumAtVtx'] = a['ele_trackMomentumAtVtx']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumAtCalo'] = a['ele_trackMomentumAtCalo']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumOut'] = a['ele_trackMomentumOut']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumAtEleClus'] = a['ele_trackMomentumAtEleClus']/a['ele_pt']
a_preprocessed['ele']['rel_trackMomentumAtVtxWithConstraint'] = a['ele_trackMomentumAtVtxWithConstraint']/a['ele_pt']
a_preprocessed['ele']['rel_ecalEnergy'] = a['ele_ecalEnergy']/a['ele_pt']
a_preprocessed['ele']['ecalEnergy_sig'] = a['ele_ecalEnergy']/a['ele_ecalEnergy_error']
a_preprocessed['ele']['rel_gsfTrack_pt'] = a['ele_gsfTrack_pt']/a['ele_pt']
a_preprocessed['ele']['gsfTrack_pt_sig'] = a['ele_gsfTrack_pt']/a['ele_gsfTrack_pt_error']

ele_has_closestCtfTrack = (a['ele_closestCtfTrack_normalizedChi2'] >= 0)
a_preprocessed['ele']['has_closestCtfTrack'] = ak.values_astype(ele_has_closestCtfTrack, np.float32)
a_preprocessed['ele']['closestCtfTrack_normalizedChi2'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_normalizedChi2'], 0)
a_preprocessed['ele']['closestCtfTrack_numberOfValidHits'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_numberOfValidHits'], 0)

ele_valid = is_finite(a['e5x5'])
ele_features = ['sigmaEtaEta', 'sigmaIetaIeta', 'sigmaIphiIphi', 'sigmaIetaIphi', 'e1x5', 'e2x5Max', 'e5x5', 'r9', 'hcalDepth1OverEcal', 'hcalDepth2OverEcal', 'hcalDepth1OverEcalBc', 'hcalDepth2OverEcalBc','eLeft', 'eRight', 'eBottom', 'eTop','full5x5_sigmaEtaEta', 'full5x5_sigmaIetaIeta', 'full5x5_sigmaIphiIphi', 'full5x5_sigmaIetaIphi','full5x5_e1x5', 'full5x5_e2x5Max', 'full5x5_e5x5', 'full5x5_r9','full5x5_hcalDepth1OverEcal', 'full5x5_hcalDepth2OverEcal', 'full5x5_hcalDepth1OverEcalBc', 'full5x5_hcalDepth2OverEcalBc','full5x5_eLeft', 'full5x5_eRight', 'full5x5_eBottom', 'full5x5_eTop','full5x5_e2x5Left', 'full5x5_e2x5Right', 'full5x5_e2x5Bottom', 'full5x5_e2x5Top']
for ele_feature in ele_features:
    _a = a[f'ele_{ele_feature}']
    a_preprocessed['ele'][ele_feature] = ak.where(_a > -1, _a, -1)

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

muon_dxy_sig_finite = is_finite(np.abs(a['muon_dxy'])) & is_finite(a['muon_dxy_error'])
a_preprocessed['muon']['dxy_sig'] = ak.where(muon_dxy_sig_finite, (np.abs(a['muon_dxy'])/a['muon_dxy_error']), 0)

muon_normalizedChi2_valid = ((a['muon_normalizedChi2'] > 0) * is_finite(a['muon_normalizedChi2']))
a_preprocessed['muon']['normalizedChi2_valid'] = ak.values_astype(muon_normalizedChi2_valid, np.float32)
a_preprocessed['muon']['normalizedChi2'] = ak.where(muon_normalizedChi2_valid, a['muon_normalizedChi2'], 0)
a_preprocessed['muon']['numberOfValidHits'] = ak.where(muon_normalizedChi2_valid, a['muon_numberOfValidHits'], 0)

muon_pfEcalEnergy_valid = is_finite(a['muon_pfEcalEnergy']) & (a['muon_pfEcalEnergy'] >= 0)
a_preprocessed['muon']['pfEcalEnergy_valid'] = ak.values_astype(muon_pfEcalEnergy_valid, np.float32)
a_preprocessed['muon']['rel_pfEcalEnergy'] = ak.where(muon_pfEcalEnergy_valid, (a['muon_pfEcalEnergy']/a['muon_pt']), 0)

add_columns = {_f: a[_f] for _f in add_feature_names} if add_feature_names is not None else None

# Check for -max_value
# for key_i in a.fields:
#     # print(key_i)
#     tmp=dask.compute(a[key_i])
#     try: 
#         tmp_=np.sum(np.isfinite(np.cosh(tmp)))
#         tmp__=80480-tmp_
#         if tmp_ != 80480 and tmp_ != 0:
#             print(f"Missing values in {key_i}: {tmp__}")
#     except:
#         pass

# a_done = dask.compute(a_preprocessed)

print("Done") #a_done)