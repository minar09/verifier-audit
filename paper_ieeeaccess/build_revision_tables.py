"""Format the audited frozen-prediction analysis; called by build_tables.py."""
import statistics as st

def build(d,out):
    def write(name,rows): (out/(name+'.tex')).write_text('\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n')
    def fmt(x):return '--' if x is None else f'{x:.3f}'
    def avg(xs):return fmt(st.mean(xs))
    def ms(xs):return avg(xs)+r' $\pm$ '+fmt(st.stdev(xs))
    def esc(x):return x.replace('_',r'\_')
    pretty={'structured':'Structured','describe':'Describe'}
    methods={'qwen3vl2b':'Qwen3-VL-2B','qwen3vl8b':'Qwen3-VL-8B','omniverifier7b':'OmniVerifier-7B'}
    protocols={'V1':'Holistic','V2':'Solve','V3':'Per option (A only)','V4':'Binary','OV':'Native verifier'}
    rows=[]
    for key,m in d['rules'].items():
        rows.append(['Pipeline validator' if key=='R0_validator' else 'Strengthened rules','Specification access',str(m['n'])]+[fmt(m[k]) for k in ('balanced_accuracy','invalid_recall','invalid_precision','valid_retention','accuracy')])
    for key,m in d['pilot_grid'].items():
        model,pr=key.split(':');rows.append([methods[model],protocols[pr],str(m['n'])]+[fmt(m[k]) for k in ('balanced_accuracy','invalid_recall','invalid_precision','valid_retention','accuracy')])
    write('pilot_grid',rows)
    rows=[]
    sets={'natural_pilot':'Pilot (development)','procgen_v3_test_id':'Assets / wording','procgen_v3_test_ood':'Two templates'}
    for ds,label in sets.items():
        for sup in pretty:
            mm=[d['internal'][ds][f'v3_{sup}_s{k}']['overall'] for k in range(3)]
            rows.append([label,pretty[sup]]+[ms([m[k] for m in mm]) for k in ('balanced_accuracy','invalid_recall','invalid_precision','valid_retention','accuracy')])
    write('internal_detailed',rows)
    tlabels={'A4_rotonly':'Rotation-only choice','B_add':'Addition detection'}
    rows=[]
    for template in ('A4_rotonly','B_add'):
        for sup in pretty:
            by=[d['internal']['procgen_v3_test_ood'][f'v3_{sup}_s{k}']['by_template'] for k in range(3)]
            if template not in by[0]:continue
            mm=[b[template] for b in by];rows.append([tlabels.get(template,esc(template)),pretty[sup],f"{mm[0]['n']:,}".replace(',',r'{,}')]+[ms([m[k] for m in mm]) for k in ('balanced_accuracy','accuracy')])
    write('templates',rows)
    rows=[]
    for suite,contrasts in d['external_contrasts'].items():
        for key,m in contrasts.items():
            a,b=key.split('-');name={'structured':'Structured','base':'2B base','omni':'Omni common','describe':'Describe'}
            rows.append([suite.replace('mm_upd','MM-UPD').replace('muir','Muir-V2').replace('dare','DARE'),name[a]+' $-$ '+name[b],f"{100*m['delta_bacc']:+.2f}",f"[{100*m['ci95'][0]:+.2f}, {100*m['ci95'][1]:+.2f}]"])
    write('paired_contrasts',rows)
    rows=[]
    for suite,models in d['native_conditional'].items():
        for sup in ('base','structured','describe'):
            mm=[models['qwen3vl2b_base']] if sup=='base' else [models[f'v3_{sup}_s{k}'] for k in range(3)]
            rows.append([suite.replace('mm_upd','MM-UPD').upper(),pretty.get(sup,'2B base'),avg([m['parsed_n']/m['n'] for m in mm]),avg([m['accuracy_all'] for m in mm]),avg([m['accuracy_given_parsed'] for m in mm])])
    write('conditional',rows)
    obs_labels={'break_key':'Break keyed option','dup_distractor':'Copy one distractor','dup_key':'Copy keyed option','invisible_literal':'Same-value distractor','key_swap':'Move key to distractor','orig':'Original item','perm':'Permute options (key remapped)','qswap_angle':'Ask another angle (stale key)','qswap_angle_rekeyed':'Ask another angle (key updated)','sym_reference [sym]':'Symmetric stimulus [sym]','qswap_color':'Ask another color'}
    write('observation_detail',[[obs_labels.get(k,esc(k)),str(v['n']),str(v['exact_obs']),str(v['n']-v.get('agree',0))] for k,v in d['observations'].items()])
    rows=[]
    for suite,models in d['operating_points'].items():
        for sup in ('omni','structured','describe'):
            mm=[models['omniverifier7b_base']] if sup=='omni' else [models[f'v3_{sup}_s{k}'] for k in range(3)]
            assert all(m['status']=='applied_unchanged' for m in mm)
            rows.append([suite.replace('mm_upd','MM-UPD').replace('muir','Muir-V2').replace('dare','DARE'),pretty.get(sup,'Omni common')]+[avg([m[k] for m in mm]) for k in ('invalid_recall','invalid_precision','valid_retention')])
    write('operating',rows)
    rows=[]
    for ds,label in sets.items():
        for sup in pretty:
            vals=[d['internal'][ds][f'v3_{sup}_s{k}']['by_defect'] for k in range(3)]
            rows.append([label,pretty[sup]]+[avg([v[defect]['invalid_recall'] for v in vals]) for defect in ('no_correct','multi_correct','key_mismatch')])
    write('defects',rows)
    # Generated diagnostic fractions, so prose does not introduce unaudited numbers.
    obs=d['observations'];ints=d['interventions']
    facts={'ObservationExact':sum(v['exact_obs'] for v in obs.values()),'ObservationN':sum(v['n'] for v in obs.values()),
        'ObservationDisagree':sum(v['n']-v['agree'] for v in obs.values()),'OracleObservationCorrect':ints['oracle_correct'],
        'OracleObservationN':ints['n'],'CounterfactualBoth':ints['both_correct_on_reference_change'],'CounterfactualN':ints['changed_reference']}
    (out/'revision_facts.tex').write_text('\n'.join(r'\newcommand{'+chr(92)+k+'}{'+str(v)+'}' for k,v in facts.items())+'\n')
