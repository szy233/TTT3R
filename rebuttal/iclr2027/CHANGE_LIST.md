**Revised PDF: list of changes.** Changed text is in blue. Numbers refer to the revised PDF. A new table in §5.2 shifts later tables by one, so the γ sweep (old Table 5) is now Table 6. The title, method and default configuration are unchanged.

**Corrections**

1. **Brake temperature.** τ=1 throughout (§4.2, §5, A.2, A.8). The submitted text printed τ=2.
2. **KITTI Odometry.** The KITTI column of Table 3 and the KITTI entries of Tables 6 and 7 now come from one rerun of all methods under one protocol (ATE RMSE, TTSA3R added). The submitted column printed ATE mean from a different set of runs, and the const entries of old Table 5 used α=0.33. "Best aggregate for outdoor driving" is withdrawn (§5.3).
3. **γ (Eq. 12, A.3, A.4, Table 6 caption).** Eq. 12 now uses the per-token EMA of drift energy that the code uses. Increasing γ moves toward ortho, and γ=0 is an off switch that runs ortho, so the family is continuous only for γ>0. TUM at γ=2 is corrected to 0.056, and a γ=5 row is added.
4. **EMA timing (Eq. 7, Algorithm 1, Figure 2(c) drawing and caption).** The direction is updated and renormalized before projection, and first-frame handling is stated.
5. **Brake and the update rule (abstract, §1, §4, A.7, A.8).** "Strictly generalizes" is removed. The brake is a related scalar baseline, not a special case. The A.8 setup with β_ema=0 was wrong and now uses an explicit previous-delta reference.
6. **Gate collapse (abstract, §1, §3.2, §4.1, Figure 1).** Narrowed to frame-averaged gates. "No meaningful temporal adaptivity" and "rules out a useful timing role" are withdrawn. The within-frame spread of TTT3R's gate is corrected to σ≈0.05.
7. **Drift energy (§3.3, §5.1, A.4, A.9, A.10, Table 7, Figure 4 caption).** It is no longer claimed to determine the sign of the direction effect or the right operating point. The two drift-energy quantities used in the paper are distinguished.
8. **Recommendations (§5.3).** Fixed scalar dampening (brake τ=1 or const 0.33, both set on these benchmarks) is the recommended default, and ortho is scoped to TUM.
9. **Smaller fixes.** Figure 1(a) now labels the brake trajectory as DDD3R_brake. TTSA3R vs. TTT3R significance (§3.2), ARKitScenes p-values (A.10), memory and throughput claims (§4, §5.3, A.1, A.6), qualitative captions (A.5), the AI use and reproducibility statements.

**Additions**

10. **§5.2 and Table 5: direction vs. step length.** Same-norm replay of ortho's per-token update norm on TUM, ScanNet, ARKitScenes and KITTI Odometry, plus a fixed-step design that matches constant dampening's step length at α=0.10, 0.20 and 0.33.
11. **A.14: the full results.** Paired tests and secondary metrics, a sweep of constant factors against ortho, rotation error over 100-frame gaps, ARKitScenes at 1000 frames, α∥=0.15, and projection onto d_{t-1} vs. the updated d_t.
12. **A.12** statistical protocol and measured numerical noise. **A.13** frame-to-frame RPE for all methods on four datasets. **A.15** per-token gate statistics and the frozen-gate test. **A.16** per-sequence TUM comparison with nested leave-one-sequence-out selection.
