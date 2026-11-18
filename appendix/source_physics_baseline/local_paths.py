"""Resolve all project inputs and results inside this checkout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'data' / 'inputfile'
RESULT_NAMES = {'FTNN':'results_10sources_5lr', 'DNN':'results_10seeds_5lr',
                'RCNN':'results_10sources_5lr_matched_seed', 'MFDNN':'results_10seeds_5lr',
                'LC':'results_10sources_lc'}

def method_paths(model):
    folder = ROOT / (model + '-use-flightdata_split')
    verified = ROOT / 'reproduction_tf215' / model
    if model in ('FTNN', 'MFDNN') and (verified / 'results' / 'experiment_config.json').is_file():
        return folder, verified / 'results', verified / 'heldout'
    return folder, folder / RESULT_NAMES[model], folder / 'heldout_tracks_4_15'

def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()
