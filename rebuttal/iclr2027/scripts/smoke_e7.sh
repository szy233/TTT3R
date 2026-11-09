#!/bin/bash
# Checks 1 and 2 of docs/PREREG_E7.md, on KITTI sequence 04, plus plumbing for the new arm.
# These ATEs are for parameter and plumbing checks only, and no verdict uses them.
# Run from the repository root.
RUN=rebuttal/iclr2027/scripts/run_arm.sh
TAG=smoke_e7
TUM=rgbd_dataset_freiburg3_walking_static

(
    export DDD3R_PORT_OFFSET=50
    bash "$RUN" 1 kitti_odo_full ortho_check "$TAG" 04
    bash "$RUN" 1 kitti_odo_full iso033 "$TAG" 04
) &
(
    export DDD3R_PORT_OFFSET=60
    bash "$RUN" 1 kitti_odo_full const033 "$TAG" 04
    bash "$RUN" 1 kitti_odo_full dir033 "$TAG" 04
) &
(
    export DDD3R_PORT_OFFSET=70
    bash "$RUN" 0 tum_s1_1000 dir033 "$TAG" "$TUM"
) &
wait
