# Correctness verification: PURE SOFT forward-KL JSD distillation loss.
# Compare loss + grad_input(student hidden) + grad_weight(lm_head) for:
#   liger_jsd (Triton), chunk_jsd eager (repo), chunk_jsd compiled (repo)
# against an independent fp32 naive reference. Uses a peaked distribution.
import os
import sys, torch
import torch.nn.functional as F
THIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_common"))
if THIS_DIR not in sys.path: sys.path.insert(0, THIS_DIR)
from jsd_kernel import fused_linear_jsd_fp32_softmax
from liger_kernel.transformers.functional import liger_fused_linear_jsd as liger_jsd

DEV="cuda"; H=4096; V=129280; BT=8192; IGNORE=-100; CHUNK=4096; BETA=0.0; TEMP=1.0
g=torch.Generator(device=DEV).manual_seed(7)

# Peaked, NON-uniform distributions: weight std chosen so logit std ~ a few.
WSTD=0.04
s_in0=torch.randn(BT,H,device=DEV,dtype=torch.bfloat16,generator=g)
t_in0=torch.randn(BT,H,device=DEV,dtype=torch.bfloat16,generator=g)
s_w0=(torch.randn(V,H,device=DEV,dtype=torch.bfloat16,generator=g)*WSTD)
t_w0=(torch.randn(V,H,device=DEV,dtype=torch.bfloat16,generator=g)*WSTD)
labels=torch.randint(0,V,(BT,),device=DEV,generator=g)   # all valid (none == IGNORE)

# sanity: how peaked is the teacher? (max softmax prob)
with torch.no_grad():
    tl_probe=(t_in0.float()@t_w0.float().t())
    p=tl_probe.softmax(-1)
    print(f"# teacher dist: mean max-prob={p.max(-1).values.mean().item():.4f} "
          f"entropy={(-(p*p.clamp_min(1e-12).log()).sum(-1)).mean().item():.3f} (uniform entropy={torch.log(torch.tensor(float(V))):.3f})")

def fresh(dtype):
    si=s_in0.detach().to(dtype).clone().requires_grad_(True)
    sw=s_w0.detach().to(dtype).clone().requires_grad_(True)
    ti=t_in0.detach().to(dtype).clone()
    tw=t_w0.detach().to(dtype).clone()
    return si,sw,ti,tw

def reference_fp32():
    si,sw,ti,tw=fresh(torch.float32)
    sl=si@sw.t()
    with torch.no_grad():
        tl=ti@tw.t()
    slp=F.log_softmax(sl,dim=-1); tlp=F.log_softmax(tl,dim=-1)
    # forward KL = KL(teacher||student) = sum_v p_t (log p_t - log p_s)
    kl=F.kl_div(slp,tlp,reduction="none",log_target=True).sum(-1)
    kl=kl.masked_fill(labels==IGNORE,0.0)
    nvalid=(labels!=IGNORE).sum().clamp_min(1)
    loss=kl.sum()/nvalid
    loss.backward()
    return loss.detach(), si.grad.detach(), sw.grad.detach()

def run_liger():
    si,sw,ti,tw=fresh(torch.bfloat16)
    loss=liger_jsd(si,sw,ti,tw,shift_labels=labels,jsd_beta=BETA,ignore_index=IGNORE,temperature=TEMP)
    loss.backward()
    return loss.detach().float(), si.grad.detach().float(), sw.grad.detach().float()

def run_chunk(compiled):
    si,sw,ti,tw=fresh(torch.bfloat16)
    loss=fused_linear_jsd_fp32_softmax(si,sw,ti,tw,labels,weight_hard_loss=0.0,weight_soft_loss=1.0,
        beta=BETA,ignore_index=IGNORE,temperature=TEMP,compiled=compiled,chunk_size=CHUNK,compute_ce_loss=False)
    loss.backward()
    return loss.detach().float(), si.grad.detach().float(), sw.grad.detach().float()

def cmp(name, L,gi,gw, Lr,gir,gwr):
    lrel=abs((L-Lr)/Lr).item()
    def stats(a,b):
        d=(a-b); rel=(d.norm()/b.norm().clamp_min(1e-12)).item()
        cos=F.cosine_similarity(a.flatten().float().unsqueeze(0),b.flatten().float().unsqueeze(0)).item()
        return d.abs().max().item(), rel, cos
    gi_mae,gi_rel,gi_cos=stats(gi,gir)
    gw_mae,gw_rel,gw_cos=stats(gw,gwr)
    print(f"{name:22s} loss={L.item():.5f} (ref {Lr.item():.5f}, rel {lrel:.2e}) | "
          f"grad_in rel {gi_rel:.2e} cos {gi_cos:.6f} maxabs {gi_mae:.2e} | "
          f"grad_w rel {gw_rel:.2e} cos {gw_cos:.6f} maxabs {gw_mae:.2e}")

print(f"# BT={BT} H={H} V={V} beta={BETA}(forward-KL) soft-only, bf16 GEMM, chunk={CHUNK}")
Lr,gir,gwr=reference_fp32()
print(f"# fp32 reference: loss={Lr.item():.5f}")
print("# (vs fp32 ref; bf16-GEMM noise ~1e-2 expected; grad cosine should be ~1.0)")
for name,fn in [("liger_jsd(Triton)",lambda:run_liger()),
                ("chunk_jsd eager",lambda:run_chunk(False)),
                ("chunk_jsd compiled",lambda:run_chunk(True))]:
    L,gi,gw=fn(); cmp(name,L,gi,gw,Lr,gir,gwr)

# KEY: compiled vs eager must be ~identical (same math). Recompute both, diff directly.
Le,gie,gwe=run_chunk(False)
Lc,gic,gwc=run_chunk(True)
print("\n# KEY compile-correctness: compiled vs eager (should be ~0):")
print(f"  loss   |diff|={abs((Lc-Le)).item():.3e}")
print(f"  grad_in  rel={(gic-gie).norm()/gie.norm():.3e}  cos={F.cosine_similarity(gic.flatten().unsqueeze(0),gie.flatten().unsqueeze(0)).item():.8f}")
print(f"  grad_w   rel={(gwc-gwe).norm()/gwe.norm():.3e}  cos={F.cosine_similarity(gwc.flatten().unsqueeze(0),gwe.flatten().unsqueeze(0)).item():.8f}")
