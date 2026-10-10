import os, json, glob, re
import numpy as np, torch, torch.nn.functional as F
import pyarrow as pa, pyarrow.parquet as pq
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel

THRESH = 0.87
OUTDIR = 'distill_gen/problems'
MODEL = "Qwen/Qwen3-Embedding-0.6B"; DEV = "cuda:0"; MAXLEN = 1024
INSTR = "Instruct: Given a competition math problem, retrieve the duplicate or semantically equivalent problem.\nQuery:"

# ---------------- load FP unique (data config = 1 row/problem) ----------------
fp = []
seen = set()
for x in sorted(glob.glob('datasets/FineProofs-SFT/data/*.parquet')):
    for r in pq.read_table(x, columns=['problem','category','competition','source',
                                        'gemini-3-pro-grade','qwen3-4b-thinking-reward@128']).to_pylist():
        if r['problem'] in seen: continue
        seen.add(r['problem']); fp.append(r)

# ---------------- load NM, dedup by problem, count traces ----------------
nm_map = OrderedDict()
with open('datasets/Nemotron-Math-Proofs-v2/data/train.jsonl','rb') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line); p = d['problem']
        if p not in nm_map:
            nm_map[p] = {'problem': p, 'uuid': d.get('uuid'), 'source': d.get('source'),
                         'dataset': d.get('dataset'), 'n_traces': 0}
        nm_map[p]['n_traces'] += 1
nm = list(nm_map.values())
print(f"FP unique={len(fp)}  NM unique={len(nm)}", flush=True)

# ---------------- embed (cache to /tmp) ----------------
def last_token_pool(h, mask):
    if mask[:, -1].sum() == mask.shape[0]:
        return h[:, -1]
    seqlens = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), seqlens]

def compute(texts, is_query, tok, model, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = texts[i:i+bs]
            if is_query: b = [f"{INSTR} {t}" for t in b]
            enc = tok(b, padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt').to(DEV)
            e = last_token_pool(model(**enc).last_hidden_state, enc['attention_mask'])
            out.append(F.normalize(e, p=2, dim=1).float().cpu())
    return torch.cat(out).numpy()

if os.path.exists('/tmp/fp_e.npy') and os.path.exists('/tmp/nm_e.npy'):
    fp_e = np.load('/tmp/fp_e.npy'); nm_e = np.load('/tmp/nm_e.npy')
    assert fp_e.shape[0] == len(fp) and nm_e.shape[0] == len(nm), "cache stale"
    print("loaded cached embeddings", flush=True)
else:
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    print("embedding NM...", flush=True); nm_e = compute([r['problem'] for r in nm], False, tok, model)
    print("embedding FP...", flush=True); fp_e = compute([r['problem'] for r in fp], True, tok, model)
    np.save('/tmp/fp_e.npy', fp_e); np.save('/tmp/nm_e.npy', nm_e)

nfp, nnm = len(fp), len(nm)
fp_g = torch.tensor(fp_e, device=DEV); nm_g = torch.tensor(nm_e, device=DEV)

# ---------------- anchor calibration ----------------
def agg(p):
    p=p.lower(); p=re.sub(r'[\\$\{\}]','',p); p=re.sub(r'[^a-z0-9]+',' ',p); return ' '.join(p.split()).strip()
nm_agg = {agg(r['problem']): j for j,r in enumerate(nm)}

# ---------------- edges ----------------
edges = []  # (node_a, node_b, sim); FP node = i ; NM node = nfp + j
# cross FP->NM
anchor_sims = []
for i in range(0, nfp, 256):
    sim = fp_g[i:i+256] @ nm_g.T
    mx, ix = sim.max(dim=1)
    for k in range(mx.shape[0]):
        gi = i + k; j = int(ix[k]); s = float(mx[k])
        if agg(fp[gi]['problem']) in nm_agg:  # anchor
            anchor_sims.append(s)
        if s >= THRESH:
            edges.append((gi, nfp + j, s))
print(f"\n=== ANCHORS (n={len(anchor_sims)}) cross-sim: min={min(anchor_sims):.3f} "
      f"max={max(anchor_sims):.3f} mean={np.mean(anchor_sims):.3f} ===", flush=True)
cross_edges = len(edges)

# intra-source: SAFE string-based only (aggressive normalization catches latex-spacing
# variants the exact match missed). Embedding self-sim over-merges distinct-but-similar
# olympiad problems (e.g. a^2+b^2+c^2=3 vs a+b+c=3 score 0.96) so it is NOT used intra.
def intra_string(records, base, label, samples=5):
    by = {}; cnt = 0; shown = 0
    for gi, r in enumerate(records):
        k = agg(r['problem'])
        if k in by:
            edges.append((base + by[k], base + gi, 1.0)); cnt += 1
            if shown < samples:
                print(f"  [{label}] A:{records[by[k]]['problem'][:80]!r}"); print(f"        B:{r['problem'][:80]!r}"); shown += 1
        else:
            by[k] = gi
    return cnt

print("\n=== intra-FP string near-dups ===", flush=True)
intra_fp = intra_string(fp, 0, 'FP')
print("\n=== intra-NM string near-dups ===", flush=True)
intra_nm = intra_string(nm, nfp, 'NM')
print(f"\nedges: cross={cross_edges} intra_fp={intra_fp} intra_nm={intra_nm}", flush=True)

# ---------------- union-find ----------------
parent = list(range(nfp + nnm))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for a, b, s in edges:
    union(a, b)
clusters = {}
for x in range(nfp + nnm):
    clusters.setdefault(find(x), []).append(x)
# per-cluster max edge sim
cl_maxsim = {}
for a, b, s in edges:
    r = find(a)
    cl_maxsim[r] = max(cl_maxsim.get(r, 0.0), s)

# ---------------- build unique rows ----------------
def fp_meta(i): return fp[i]
def nm_meta(j): return nm[j - nfp]
rows = []
for root, members in clusters.items():
    fp_ms = [m for m in members if m < nfp]
    nm_ms = [m - nfp for m in members if m >= nfp]
    origin = 'both' if (fp_ms and nm_ms) else ('FineProofs' if fp_ms else 'Nemotron-Math-Proofs-v2')
    # representative = longest problem text
    cand = [(fp[m]['problem'], 'FineProofs') for m in fp_ms] + [(nm[m]['problem'], 'Nemotron-Math-Proofs-v2') for m in nm_ms]
    rep_text, rep_origin = max(cand, key=lambda t: len(t[0]))
    fpm = fp[fp_ms[0]] if fp_ms else None
    nmm = nm[nm_ms[0]] if nm_ms else None
    members_json = []
    for m in fp_ms:
        members_json.append({'origin':'FineProofs','source':fp[m]['source'],'category':fp[m]['category'],
                             'competition':fp[m]['competition'],'gemini_grade':fp[m]['gemini-3-pro-grade'],
                             'problem':fp[m]['problem']})
    for m in nm_ms:
        members_json.append({'origin':'Nemotron-Math-Proofs-v2','source':nm[m]['source'],'uuid':nm[m]['uuid'],
                             'n_traces':nm[m]['n_traces'],'problem':nm[m]['problem']})
    rows.append({
        'problem': rep_text,
        'origin': origin,
        'rep_source': rep_origin,
        'category': fpm['category'] if fpm else None,
        'competition': fpm['competition'] if fpm else None,
        'source': (fpm['source'] if fpm else None) or (nmm['source'] if nmm else None),
        'fp_gemini_grade': fpm['gemini-3-pro-grade'] if fpm else None,
        'fp_qwen_reward': fpm['qwen3-4b-thinking-reward@128'] if fpm else None,
        'nm_uuid': nmm['uuid'] if nmm else None,
        'nm_n_traces': nmm['n_traces'] if nmm else None,
        'n_members': len(members),
        'merge_max_cosine': round(cl_maxsim.get(root, 1.0), 4) if len(members) > 1 else None,
        'members': json.dumps(members_json, ensure_ascii=False),
    })

# stats
from collections import Counter
oc = Counter(r['origin'] for r in rows)
merged = [r for r in rows if r['n_members'] > 1]
print(f"\n=== RESULT ===")
print(f"total unique problems: {len(rows)}")
print(f"origin breakdown: {dict(oc)}")
print(f"merged clusters (>1 member): {len(merged)}  | size dist: {Counter(r['n_members'] for r in merged)}")
print(f"raw input total: {nfp + nnm}  -> merged away: {nfp + nnm - len(rows)}")

# ---------------- write ----------------
os.makedirs(OUTDIR, exist_ok=True)
tbl = pa.Table.from_pylist(rows)
pq.write_table(tbl, f'{OUTDIR}/problems.parquet', compression='zstd')

# read-back verify (host bit-flip safety)
back = pq.read_table(f'{OUTDIR}/problems.parquet')
assert back.num_rows == len(rows), "row count mismatch on readback"
bp = set(back.column('problem').to_pylist()); op = set(r['problem'] for r in rows)
assert bp == op, "problem set mismatch on readback"
print(f"\nwrote {OUTDIR}/problems.parquet  ({back.num_rows} rows) -- readback verified OK")

# save stats json for the README step
json.dump({'total':len(rows),'origin':dict(oc),'merged':len(merged),
           'cross_edges':cross_edges,'intra_fp':intra_fp,'intra_nm':intra_nm,
           'nfp':nfp,'nnm':nnm,'thresh':THRESH,
           'anchor_min':min(anchor_sims),'anchor_max':max(anchor_sims)},
          open('/tmp/dedup_stats.json','w'), indent=2)
