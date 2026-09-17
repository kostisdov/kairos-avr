import pandas as pd, re, collections
from kairos.paths import repo_root
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_colwidth', 60)
D = f"{repo_root()}/"
notes = pd.read_excel(D+"notes_deidentified.xlsx")
labs = pd.read_excel(D+"labs_deidentified.xlsx")
notes = notes[notes['Signed Status'] != 'Deleted'].copy()

models = {
 'SAPIEN': r'sapien', 'Evolut/CoreValve': r'evolut|corevalve|corevalue', 'Navitor/Portico': r'navitor|portico',
 'Perimount/Magna/CE': r'perimount|magna|carpentier|#\s?\d\d\s?CE\b', 'Inspiris Resilia': r'inspiris|resilia',
 'Trifecta': r'trifecta', 'Konect(BioBentall)': r'konect',
 'porcine(Hancock/Mosaic/Freestyle/Epic-valve)': r'hancock|mosaic|freestyle|\bepic (?:valve|bioprosth|supra|stented)',
 'Avalus': r'avalus', 'Intuity': r'intuity', 'Perceval': r'perceval', 'Mitroflow': r'mitroflow',
}
tavr_pat = r'transcatheter aortic valve|\bTAVR\b|\bTAVI\b|sapien|evolut|corevalve|corevalue'
savr_pat = r'aortic valve replacement with|\bAVR\b\s*\(?#|sternotomy|bioprosthesis via|tissue implant type|replacement type: tissue'
size_pats = [r'#\s?(\d{2})(?:\s?-?\s?mm)?\s?(?:trifecta|magna|perimount|CE\b|inspiris|konect|valve|bioprosth)',
             r'(\d{2})\s?-?\s?mm\s+(?:trifecta|magna|perimount|inspiris|edwards|sapien|evolut|bioprosthesis|valve)',
             r'implant size:\s*(\d{2})', r'size:\s*(?:ultra\s*|R)?(\d{2})\s?mm', r'size #\s?(\d{2})', r'\(#(\d{2})\)']
mg_pats = [r'mean gradient (?:is|of|was|=)?\s*(\d{1,3})\s*mm\s?hg', r'peak/mean gradients?\s*(?:of|were|was|:)?\s*\d{1,3}/(\d{1,3})',
           r'gradients?:?\s*(?:of\s*)?\d{1,3}/(\d{1,3})\s*mm\s?hg', r'\bMG\s*(?:of|is|=)?\s*(\d{1,3})\s*mm', r'mean (?:PG|pressure gradient)[^0-9]{0,12}(\d{1,3})']
ev = {
 'ViV': r'valve[- ]in[- ]valve|\bViV\b',
 'redo': r'\bredo\b',
 'SVD': r'structural valve deterioration|\bSVD\b|prosthetic (?:aortic )?valve (?:dysfunction|failure|stenosis|regurgitation)|bioprosthetic (?:AVR|aortic valve|valve) (?:stenosis|dysfunction|failure|degeneration)|prosthetic thickening|degenerat\w+ (?:bio)?prosth|T82\.0|failed (?:bio)?prosth',
 'endoc': r'endocarditis', 'thromb': r'valve thromb|leaflet thromb|\bHALT\b',
}
rows = []
for _, r in notes.iterrows():
    t = str(r['Notes']); pid = r['Profile Key']; yr = int(r['Service Date']); typ = r['Type']
    found_models = [k for k, p in models.items() if re.search(p, t, re.I)]
    is_tavr = bool(re.search(tavr_pat, t, re.I)); is_savr = bool(re.search(savr_pat, t, re.I))
    sizes = set()
    for p in size_pats:
        for m in re.finditer(p, t, re.I):
            v = int(m.group(1))
            if 17 <= v <= 34: sizes.add(v)
    mgs = []
    for p in mg_pats:
        for m in re.finditer(p, t, re.I):
            v = int(m.group(1))
            if 0 < v < 120:
                ctx = t[max(0, m.start()-260):m.end()+60]
                prosth = bool(re.search(r'prosthetic|sapien|evolut|bioprosth|trifecta|magna|perimount|s/p|status post|post[- ]?op|TAVR|AVR', ctx, re.I))
                native = bool(re.search(r'native|calcified valve|caused by calcified|bicuspid|pre-?op|preoperative', ctx, re.I)) and not re.search(r'prosthetic thick', ctx, re.I)
                mgs.append((v, prosth, native))
    events = {k: bool(re.search(p, t, re.I)) for k, p in ev.items()}
    rows.append(dict(pid=pid, yr=yr, typ=typ, models=found_models, tavr=is_tavr, savr=is_savr, sizes=sizes, mgs=mgs, **events))
df = pd.DataFrame(rows)
df['implant_note'] = df['typ'].isin(['Operative Report', 'Procedures']) & (df['tavr'] | df['savr'])

per = []
for pid, g in df.groupby('pid'):
    imp = g[g['implant_note']]
    implant_years = sorted(imp['yr'].unique().tolist())
    mods = sorted({m for L in g['models'] for m in L})
    sizes = sorted({s for S in g['sizes'] for s in S})
    if len(imp):
        has_t = imp['tavr'].any(); has_s = (imp['savr'] & ~imp['tavr']).any()
        route = ('TAVR' if has_t else '') + ('+' if has_t and has_s else '') + ('SAVR' if has_s else '')
    else:
        route = '(hist only)' if (g['tavr'].any() or g['savr'].any()) else ''
    pros_mg = [(y, v) for y, L in zip(g['yr'], g['mgs']) for (v, p, n) in L if p and not n]
    any_mg = [(y, v) for y, L in zip(g['yr'], g['mgs']) for (v, p, n) in L]
    per.append(dict(pid=pid, n_notes=len(g), first=g['yr'].min(), last=g['yr'].max(), implant_years=implant_years, route=route,
        models=','.join(mods), sizes=sizes, n_mg_any=len(any_mg), n_mg_prosth=len(pros_mg), mg_years=sorted({y for y, _ in pros_mg}),
        max_prosth_mg=max([v for _, v in pros_mg], default=None),
        ViV=g['ViV'].any(), redo=g['redo'].any(), SVD=g['SVD'].any(), endoc=g['endoc'].any(), thromb=g['thromb'].any()))
P = pd.DataFrame(per)
P['fu_years'] = P['last'] - P['first']
has_imp = P['implant_years'].str.len() > 0
print("patients:", len(P))
print("with implant op/procedure note:", int(has_imp.sum()), " route counts:", P['route'].value_counts().to_dict())
print("with any valve model named:", int((P['models'] != '').sum()))
print("model frequency (patients):", collections.Counter(m for s in P['models'] for m in s.split(',') if m))
print("with valve size:", int((P['sizes'].str.len() > 0).sum()), " size dist:", dict(sorted(collections.Counter(s for S in P['sizes'] for s in S).items())))
print("with >=1 prosthetic mean gradient:", int((P['n_mg_prosth'] > 0).sum()), " with >=2 distinct gradient years:", int((P['mg_years'].str.len() >= 2).sum()))
print("with any mean gradient (incl native):", int((P['n_mg_any'] > 0).sum()))
print("follow-up span (yrs) among implant-recorded pts:", P.loc[has_imp, 'fu_years'].describe().round(1).to_dict())
print("span>=1y:", int((P['fu_years'] >= 1).sum()), " span>=5y:", int((P['fu_years'] >= 5).sum()), " span>=8y:", int((P['fu_years'] >= 8).sum()))
print("events: ViV", int(P['ViV'].sum()), " redo", int(P['redo'].sum()), " SVD/dysfunction", int(P['SVD'].sum()), " endocarditis", int(P['endoc'].sum()), " thrombosis", int(P['thromb'].sum()))
print("implant + >=1 prosthetic gradient:", int((has_imp & (P['n_mg_prosth'] > 0)).sum()))
print("\n=== patients with event flags or >=2 gradient-years (or prosthetic MG>=20)")
sel = P[(P['ViV'] | P['redo'] | P['SVD'] | P['endoc'] | P['thromb']) | (P['mg_years'].str.len() >= 2) | (P['max_prosth_mg'].fillna(0) >= 20)]
print(sel[['pid', 'n_notes', 'first', 'last', 'implant_years', 'route', 'models', 'sizes', 'mg_years', 'max_prosth_mg', 'ViV', 'redo', 'SVD', 'endoc', 'thromb']].to_string(index=False))
print("\n=== the 17 structured-data patients")
s17 = P[P['pid'].isin(set(labs.Patient))]
print(s17[['pid', 'n_notes', 'first', 'last', 'implant_years', 'route', 'models', 'sizes', 'n_mg_prosth', 'max_prosth_mg', 'ViV', 'redo', 'SVD']].to_string(index=False))
key_labs = {'NT-proBNP': 'NT PRO BNP|PROBNP', 'HbA1c': 'HEMOGLOBIN A1C', 'eGFR': 'EGFR|GFR', 'Phosphorus': 'PHOSPHORUS', 'LDL': '^LDL$', 'Lp(a)': 'LIPOPROTEIN', 'CRP': 'C-REACTIVE|^CRP', 'LVEF': 'LV EJECTION', 'INR': 'PT INR', 'Calcium': 'CALCIUM, TOTAL', 'Troponin': 'TROPONIN'}
lab_tab = {}
up = labs['Lab Component Common Name'].astype(str).str.upper()
for k, p in key_labs.items():
    m = up.str.contains(p, regex=True)
    lab_tab[k] = labs[m].groupby('Patient').size()
LT = pd.DataFrame(lab_tab).fillna(0).astype(int)
print("\nkey lab counts per structured patient:\n", LT.to_string())
lvef = labs[up.str.contains('LV EJECTION')]
print("\nLVEF rows by patient/year (n):\n", lvef.groupby(['Patient', 'Result Date']).size().unstack(fill_value=0).to_string())
print("\n=== Procedures notes: first 400 chars of two")
for t in notes[notes['Type'] == 'Procedures']['Notes'].head(2):
    print(' '.join(str(t)[:400].split())); print('---')
