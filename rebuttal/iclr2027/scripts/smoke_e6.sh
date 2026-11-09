#!/bin/bash
# Short checks before the E6 queues, on one sequence each. Run from the repository root.
# These ATEs are for plumbing and parameter checks only, and no verdict uses them.
RUN=rebuttal/iclr2027/scripts/run_arm.sh
TUM=rgbd_dataset_freiburg3_walking_static

(
    export DDD3R_PORT_OFFSET=30
    bash "$RUN" 0 tum_s1_1000 ortho_a015 smoke "$TUM"
    bash "$RUN" 0 tum_s1_1000 normreplay_a015 smoke "$TUM"
    bash "$RUN" 0 tum_s1_1000 const033 smoke "$TUM"
) &
(
    export DDD3R_PORT_OFFSET=40
    bash "$RUN" 1 kitti_odo_full ortho_check smoke 04
    bash "$RUN" 1 kitti_odo_full normreplay smoke 04
) &
wait
