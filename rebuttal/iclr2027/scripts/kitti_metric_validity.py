"""Is r_rel a valid measure here? Score a trajectory with ground-truth positions and a frozen
orientation, and measure how much each arm's predicted trajectory rotates. Runs no model.

Usage, from the repository root:
    PYTHONPATH=src:. python rebuttal/iclr2027/scripts/kitti_metric_validity.py > rebuttal/iclr2027/results/kitti_metric_validity.txt
"""
import os,sys
import numpy as np
sys.path.insert(0,'.')
from eval.relpose.evo_utils import load_replica_traj, kitti_odom_eval_from_tum, make_traj
from eval.relpose.eval_kitti_odom import load_pred_traj_tum
from scipy.spatial.transform import Rotation as R
GT='/mnt/sda/rkj/VGGT_Long_outdoor/kitti_00_10/gt_poses_kitti'
if not os.path.isdir(GT):
    import re
    src=open('eval/relpose/eval_kitti_odom.py').read()
    GT=re.search(r'"--gt_dir", type=str, default="([^"]+)"',src).group(1)
RES='rebuttal/iclr2027/results/relpose/kitti_odo_full'
seqs=[f'{i:02d}' for i in range(11)]
def quatwxyz(t): return t[:,3:7]
def score(traj,gt):
    t,r,_=kitti_odom_eval_from_tum(traj,gt)
    return t,r
def stationary_heading(gt):
    """GT positions, with the orientation frozen at the first frame: a method that never turns."""
    arr,ts=gt
    out=arr.copy(); out[:,3:7]=arr[0,3:7]
    return out,ts
def segment_rotation(arr,step=10):
    """Mean absolute rotation over a fixed frame window, in degrees."""
    q=arr[:,3:7]; rot=R.from_quat(q[:,[1,2,3,0]])
    d=(rot[:-step].inv()*rot[step:]).magnitude()
    return float(np.degrees(d).mean())
rows={}
print('seq   frozen_rotation_r_rel  gt_rot_per10f')
base=[]
gtrot={}
for s in seqs:
    gt=load_replica_traj(f'{GT}/{s}.txt')
    t,r=score(stationary_heading(gt),gt)
    base.append((t,r)); gtrot[s]=segment_rotation(gt[0])
    print(f'{s}   t_rel={t:8.2f}  r_rel={r:7.3f}   gt_rot/10f={gtrot[s]:.3f}')
print('MEAN frozen-orientation with GT positions: t_rel=%.2f r_rel=%.3f'%(np.mean([b[0] for b in base]),np.mean([b[1] for b in base])))
print()
print(f"{'arm':12s}{'r_rel':>8s}{'rot/10f, ratio to GT':>24s}")
import re
vals={}
for name,src in (('ttt3r_plain','kitti_official_metrics.txt'),('const05','kitti_official_metrics.txt'),('brake_t1','kitti_official_metrics.txt'),('const033','e6_kitti_metrics.txt'),('const020','e6_kitti_metrics.txt'),('const010','e6_kitti_metrics.txt'),('normreplay','e6_kitti_metrics.txt'),('ortho_check','e6_kitti_metrics.txt'),('orthop','e7_kitti_metrics.txt'),('dir033','e7_kitti_metrics.txt'),('dir020','e7_kitti_metrics.txt')):
    rr=[];pr=[]
    for s in seqs:
        p=f'{RES}/{name}/{s}/pred_traj.txt'
        if not os.path.exists(p): continue
        tr,_=load_pred_traj_tum(p)
        pr.append(segment_rotation(tr)/gtrot[s])
    m=re.findall(rf'^\s*{name}\s*\|\s*\d\d\s*\|\s*t_rel:\s*[0-9.]+%\s*\|\s*r_rel:\s*([0-9.]+)',open(f'rebuttal/iclr2027/results/{src}').read(),flags=re.M)
    rv=np.mean([float(x) for x in m]) if m else float('nan')
    vals[name]=(rv,np.mean(pr) if pr else float('nan'))
    print(f'{name:12s}{rv:8.2f}{"":>6s}{np.mean(pr) if pr else float("nan"):13.3f}')
