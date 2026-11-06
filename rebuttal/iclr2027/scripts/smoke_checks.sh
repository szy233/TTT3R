#!/bin/bash
# Quick checks before the full queues. Run from the repository root.
#   GPU 1: every arm on one 90-frame TUM sequence (plumbing and baseline match).
#   GPU 0: KITTI sequence 04 under both protocols (E5 check and Table 5 provenance).
RUN=rebuttal/iclr2027/scripts/run_arm.sh
SEQ=rgbd_dataset_freiburg3_walking_static

(
    for arm in ortho_check normmatch prevdir ttt3r_log ttt3r_frozen ttsa3r_log ttsa3r_frozen; do
        bash $RUN 1 tum "$arm" smoke "$SEQ"
    done
) &
(
    for arm in cut3r brake_t2; do
        bash $RUN 0 kitti_odo_full "$arm" check04 04
    done
    for arm in brake_t1 brake_t2 cut3r; do
        bash $RUN 0 kitti_odom "$arm" t5 04
    done
) &
wait
