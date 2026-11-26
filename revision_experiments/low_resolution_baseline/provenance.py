"""Bind cached training/evaluation artifacts to an immutable experiment identity."""
from pathlib import Path
import ast,json,hashlib
import torch,numpy as np

HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def numeric_digest(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    # Include every numerical/data/training function and class, including the
    # complete train_all loop. Entry-point provenance wrapping is not numerical.
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    return hashlib.sha256('\n'.join(ast.dump(n,include_attributes=False) for n in nodes).encode()).hexdigest()

def identity():
    from controlled_training import SOURCE,TEST,FINE,CODE,MODELS,SEEDS
    inputs={str(p):digest(p) for p in [SOURCE,TEST,FINE]}
    return {'numeric_implementation_sha256':numeric_digest(HERE/'controlled_training.py'),
        'design_sha256':digest(HERE/'design.txt'),'models':MODELS,'seeds':SEEDS,
        'original_model_sources':{k:digest(CODE/k) for k in ['fno.py','ufno.py','ablation_study.py','data_training.py']},
        'simulator_source_sha256':digest(CODE/'data_generation.py'),
        'timing_helper_sha256':digest(HERE.parent/'run_experiments.py'),
        'data_sha256':inputs,'torch':torch.__version__,'numpy':np.__version__}

def artifact_paths():
    files=[HERE/x for x in ['split.json','training_normalization.json','source_normalized.pt','config.json','training_summary.json']]
    files+=list(HERE.glob('*_history.json'))+list(HERE.glob('*_completed.json'))+list((HERE/'checkpoints').glob('*.pt'))
    return sorted(p for p in files if p.exists())

def seal(audited_fresh_run=False):
    meta_path=HERE/'training_provenance.json'
    current=identity()
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        assert old['identity']==current,'Experiment implementation/configuration/data changed; use a new experiment directory.'
    elif audited_fresh_run:
        snapshot=HERE/'controlled_training_source_v1.py'
        assert snapshot.exists() and numeric_digest(snapshot)==current['numeric_implementation_sha256']
        summary=json.loads((HERE/'training_summary.json').read_text());assert len(summary)==12
        norm=json.loads((HERE/'training_normalization.json').read_text())
        config=json.loads((HERE/'config.json').read_text())
        assert config['design_sha256']==current['design_sha256']
        assert config['normalization_sha256']==digest(HERE/'training_normalization.json')
        assert config['source_hashes']==current['original_model_sources']
        for r in summary:
            assert digest(r['checkpoint'])==r['sha256']
            c=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
            assert c['normalization']==norm and c['config']==config
        current['audit_note']='Fresh run observed from an empty output directory; numerical source snapshot verified identical before adding provenance-only entry-point checks.'
    paths=artifact_paths()
    # The note is provenance, not an additional identity dimension.
    note=current.pop('audit_note',None)
    result={'identity':current,'artifacts':{str(p.relative_to(HERE)):digest(p) for p in paths}}
    if note:result['audit_note']=note
    meta_path.write_text(json.dumps(result,indent=2))

def verify():
    p=HERE/'training_provenance.json';assert p.exists(),'Missing provenance seal. Refuse unverified cached results.'
    saved=json.loads(p.read_text());assert saved['identity']==identity(),'Inputs or numerical implementation changed. Start a new registered experiment.'
    actual={str(p.relative_to(HERE)) for p in artifact_paths()}
    assert actual==set(saved['artifacts']),'Unexpected or missing cached training artifacts.'
    for name,h in saved['artifacts'].items():assert digest(HERE/name)==h,'Changed training artifact '+name

def guarded_train(train):
    p=HERE/'training_provenance.json'
    if p.exists():verify()
    else:
        assert not artifact_paths(),'Unsealed existing artifacts. Refuse silent reuse; audit the observed fresh run explicitly.'
        p.write_text(json.dumps({'identity':identity(),'artifacts':{}},indent=2))
    try:train()
    finally:seal()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seal-audited-fresh-run',action='store_true');args=parser.parse_args()
    if args.seal_audited_fresh_run:seal(audited_fresh_run=True)
    verify();print('Training provenance and artifacts verified.')
