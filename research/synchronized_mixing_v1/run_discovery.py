"""SMC-05: discover anonymous sparse feature geometry, then unknown frame.

The geometry learner does not receive permutation-to-feature wiring.
The source fixture retains hidden frame/scramble only for scoring.
"""
import json
import random
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from discover_geometry import (
    infer_feature_geometry,decode_mask,noisy_replicates,
    majority_denoise,switching_null,
)
from identify_transport import uniform_catalog,hidden_fixture,predict,version_space

HOLDOUT=(12,13,14)
EDGE_LIST=tuple(combinations(range(6),2))


def fixture(seed=43,feature_seed=1701):
    """Source simulation only; geometry inference sees anonymous binary rows."""
    c=uniform_catalog()
    truth=hidden_fixture(c,seed)
    rng=random.Random(feature_seed)
    scramble=list(range(15))
    rng.shuffle(scramble)
    def encode_matching(g):
        mask=[0]*15
        for j,e in enumerate(EDGE_LIST):
            mask[scramble[j]]=int(e in g)
        return tuple(mask)
    masks=tuple(encode_matching(g) for g in c['matchings'])
    train_states=tuple(j for j in range(15) if j not in HOLDOUT)
    train_pi=tuple(i for i in range(720) if predict(c,truth,i) in train_states)
    test_pi=tuple(i for i in range(720) if predict(c,truth,i) in HOLDOUT)
    assert len(train_pi)==576 and len(test_pi)==144
    return {'catalog':c,'truth':truth,'train_states':train_states,
            'holdout_states':HOLDOUT,'train_perms':train_pi,
            'test_perms':test_pi,'masks':masks,
            '_hidden_scramble':tuple(scramble)}


def discover_from_unlabelled_train_states(fixture,noise=(1.0,0.0),
                                          repeats=1,seed=400):
    """Fit only the 12 grouped, anonymous activation-state episodes."""
    masks=tuple(fixture['masks'][j] for j in fixture['train_states'])
    episodes=noisy_replicates(masks,repeats=repeats,
                             sensitivity=noise[0],false_positive=noise[1],seed=seed)
    recovered=majority_denoise(episodes)
    return infer_feature_geometry(recovered),recovered


def evaluate_hat(fixture,geometry,hat):
    """Test on withheld graph classes only after model choice; matched negatives."""
    c=fixture['catalog']
    test=fixture['test_perms']
    mid=c['mindex']
    observed=tuple(mid[decode_mask(
       geometry,fixture['masks'][predict(c,fixture['truth'],i)])] for i in test)
    negatives=tuple(sorted(set(observed)))
    assert len(negatives)==3
    pred=tuple(predict(c,hat,i) for i in test)
    tpr=Fraction(sum(u==v for u,v in zip(pred,observed)),len(test))
    fpr=Fraction(sum(g in negatives for g in pred),len(test)*len(negatives))
    return {'tpr':tpr,'fpr':fpr,'balanced_accuracy':(1+tpr-fpr)/2,
            'test_abstract_graph_classes':negatives}


def identify(fixture,geometry,inputs):
    """Observed (input,latent-output) pairs; no secret enters version_space."""
    c=fixture['catalog']
    observations=tuple((i,c['mindex'][decode_mask(
        geometry,fixture['masks'][predict(c,fixture['truth'],i)])])
        for i in inputs)
    hypotheses=version_space(c,observations)
    if not hypotheses:
        raise AssertionError('no transform fits reconstructed geometry')
    return hypotheses,observations


def measured_candidate_source(fixture,geometry):
    """Evaluation-only oracle using all training pairs; never chooses the learner."""
    hypotheses,_=identify(fixture,geometry,fixture['train_perms'])
    assert len(hypotheses)==1
    return hypotheses[0]


def simulate(seed=43,feature_seed=1701,calibration_seed=901,examples=12,
             noise=(1.0,0.0),repeats=1):
    f=fixture(seed,feature_seed)
    learned,_=discover_from_unlabelled_train_states(
        f,noise=noise,repeats=repeats,seed=calibration_seed)
    rng=random.Random(calibration_seed+991)
    inputs=[rng.choice(f['train_perms']) for _ in range(examples)]
    hypotheses,_=identify(f,learned,inputs)
    hat=hypotheses[0]
    scores=evaluate_hat(f,learned,hat)
    true_aligned=measured_candidate_source(f,learned)
    exact_heldout=evaluate_hat(f,learned,true_aligned)
    assert exact_heldout['balanced_accuracy']==Fraction(5,6)
    return {'seed':seed,'feature_seed':feature_seed,
            'calibration_seed':calibration_seed,
            'noise_sensitivity':noise[0],
            'noise_false_positive':noise[1],
            'episode_repeats':repeats,'calibration_pairs':examples,
            'reconstructed_vertex_stars':len(learned['cliques']),
            'reconstructed_edges':len(learned['feature_edges']),
            'train_matching_states':len(f['train_states']),
            'heldout_matching_states':len(f['holdout_states']),
            'calibration_version_space':len(hypotheses),
            'inferred_model_matches_truth':hat==true_aligned,
            'heldout_balanced_accuracy':str(scores['balanced_accuracy']),
            'oracle_heldout_balanced_accuracy':str(exact_heldout['balanced_accuracy']),
            'coactive_pairs':learned['coactive_pairs'],
            'noncoactive_pairs':learned['noncoactive_pairs'],
            'candidate_5_cliques':learned['candidate_5_cliques'],
            'note':'geometry sees only 12 anonymous training-state masks'}


def geometry_noise_campaign(seeds=range(100),noise=(0.75,0.25),repeats=40):
    f=fixture()
    gtruth=discover_from_unlabelled_train_states(f)[0]
    true_stars=set(map(frozenset,gtruth['cliques']))
    accepted=correct=0
    for seed in seeds:
        try:
            g,_=discover_from_unlabelled_train_states(f,noise,repeats,seed)
            accepted+=1
            correct+=set(map(frozenset,g['cliques']))==true_stars
        except ValueError:
            pass
    return {'noise':noise,'repeats':repeats,'trials':len(seeds),
            'certified_reconstructions':accepted,
            'correct_reconstructions':correct}


def null_campaign(seeds=range(100),steps=4000):
    f=fixture()
    masks=tuple(f['masks'][j] for j in f['train_states'])
    expected_col=[sum(row[j] for row in masks) for j in range(15)]
    accepted=0
    for seed in seeds:
        rewired=switching_null(masks,seed,steps)
        assert all(sum(row)==3 for row in rewired)
        assert expected_col==[sum(row[j] for row in rewired) for j in range(15)]
        try:
            infer_feature_geometry(rewired)
            accepted+=1
        except ValueError:
            pass
    return {'null':'degree-preserving 2x2 incidence matrix switches',
            'trials':len(seeds),'geometry_certificates_accepted':accepted,
            'row_and_column_sums_exactly_preserved':True}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('smc05-results.json'))
    args=parser.parse_args()
    n=100 if args.full else 8
    report={'schema':'SMC05_anonymous_sparse_geometry_v1',
      'clean':[simulate(seed,1701+seed,seed+29,examples=12)
               for seed in (43,77,91)],
      'denoised':[simulate(43,1701,100+j,examples=12,
                           noise=(.75,.25),repeats=40) for j in range(3)],
      'geometry_noise':[geometry_noise_campaign(seeds=range(n),noise=noise,
                         repeats=rep) for noise,rep in
                         (((.9,.1),16),((.75,.25),40),((.75,.25),64))],
      'null':null_campaign(seeds=range(n)),
      'limitations':'Six vertices, three-active-edge states and S6 action after discovery remain structural priors; no pretrained transformer or SAE.'}
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))
