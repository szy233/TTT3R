# NeurIPS 2026 Submission 8591 — Hallucinated References Appeal
# Deadline: 9 September AoE. Form: https://forms.gle/HKLuPvKCjz87tsNM7
# One entry per flagged reference. 6 flagged.

===============================================================================
APPEAL 1 — MASt3R   [STRONGEST — use option (a)]
===============================================================================
Reference identifier in paper:  \cite{mast3r}  (Related Work, Sec. 2)

Exact citation as flagged:
@inproceedings{leroy2024mast3r,
  author = {Vincent Leroy and Yohann Cabon and Jérôme Revaud},
  title = {MASt3R: Matching and stereo 3d reconstruction},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}

Authoritative source for the citation format used:
https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/
(NAVER LABS Europe, the authors' own institution, publishes this work under the
title "MASt3R - Matching And Stereo 3D Reconstruction".)
Secondary: https://arxiv.org/abs/2406.09756  |  ECCV 2024, doi:10.1007/978-3-031-73220-1_5

Basis: (a) The reference checker's outcome is incorrect. I have attached a public
authoritative source for the citation format I used.

Explanation:
The author list (Vincent Leroy, Yohann Cabon, Jérôme Revaud), the venue (ECCV) and
the year (2024) are exactly correct. The only discrepancy is that our title uses the
expansion of the MASt3R acronym, "Matching And Stereo 3D Reconstruction", rather than
the proceedings title "Grounding Image Matching in 3D with MASt3R". That expansion is
not invented: it is the title under which the authors' own institution, NAVER LABS
Europe, publishes the work (link above). Under the stated criteria, "minor differences
in title should not be considered evidence of a hallucinated reference", and no author,
venue or existence issue applies here.

===============================================================================
APPEAL 2 — DUSt3R   [STRONG — use option (c)]
===============================================================================
Reference identifier in paper:  \cite{dust3r}  (Related Work, Sec. 2)

Exact citation as flagged:
@inproceedings{wang2024dust3r,
  author = {Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Raber and Jérôme Revaud},
  title = {DUSt3R: Geometric 3d vision made easy},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}

Authoritative source:
https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html
Secondary: https://arxiv.org/abs/2312.14132

Basis: (c) The reference checker's outcome is correct. The errors in my citation are
minor, such as ... minor misspellings in venue or author names.

Explanation:
The title, venue, year, author count and author order are all exactly correct, as are
four of the five author names. The single error is the surname of the fourth author,
written as "Boris Raber" instead of "Boris Chidlovskii". The given name and the position
in the author list are correct, so this is a corrupted surname rather than the addition
of a person who is not an author of the publication. Under the stated criteria, authors
that are "only slightly incorrect" do not imply a hallucinated reference. We will correct
this in the camera-ready.

===============================================================================
APPEAL 3 — TTT3R   [MEDIUM — use 其他 / Other]
===============================================================================
Reference identifier in paper:  \cite{ttt3r}  (baseline throughout)

Exact citation as flagged:
@inproceedings{ji2025ttt,
  author = {Chaoyue Ji et al.},
  title = {Ttt-3dr: Test-time training for 3d reconstruction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025}
}

Authoritative source for the cited work:
https://arxiv.org/abs/2509.26645
"TTT3R: 3D Reconstruction as Test-Time Training",
Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, Anpei Chen.

Basis: Other.

Explanation:
We do not dispute that the author field is wrong, and we correct it below. We appeal the
classification for two reasons. First, on title: the cited title "TTT-3DR: Test-Time
Training for 3D Reconstruction" carries the same acronym and the same three terms as the
real title "TTT3R: 3D Reconstruction as Test-Time Training", reordered. Under the stated
criteria this falls under "minor differences in title" rather than a title that cannot be
found. Second, on identity: TTT3R is the direct baseline of our submission, it is named in
our title, abstract and every results table, and our released code is built on it, so there
is no ambiguity about which real work is being cited. The defect is corrupted bibliographic
metadata in a placeholder .bib entry that we failed to verify before submission, not a
reference to a work that does not exist.
Correct entry: Chen, Xingyu and Chen, Yue and Xiu, Yuliang and Geiger, Andreas and
Chen, Anpei. "TTT3R: 3D Reconstruction as Test-Time Training". arXiv:2509.26645, 2025.

===============================================================================
APPEAL 4 — TTSA3R   [MEDIUM — use 其他 / Other]
===============================================================================
Reference identifier in paper:  \cite{ttsa3r}  (Sec. 3.2, Related Work)

Exact citation as flagged:
@article{zhao2025ttsa3r,
  author = {Mingwei Zhao et al.},
  title = {Ttsa3r: Temporal-spatial adaptive 3d reconstruction},
  journal = {arXiv preprint arXiv:2503.18418},
  year = {2025}
}

Authoritative source for the cited work:
https://arxiv.org/abs/2601.22615
"TTSA3R: Training-Free Temporal-Spatial Adaptive Persistent State for Streaming 3D
Reconstruction", Zhijie Zheng, Xinhao Xiang, Jiawei Zhang.

Basis: Other.

Explanation:
We do not dispute that the author field and the arXiv identifier are wrong, and we correct
them below. We appeal the classification on the title: the cited title "TTSA3R:
Temporal-Spatial Adaptive 3D Reconstruction" is an abbreviated form of the real title and
retains the acronym, the distinguishing phrase "Temporal-Spatial Adaptive", and "3D
Reconstruction". Under the stated criteria a wrong arXiv identifier is explicitly not
treated as evidence of hallucination, and the title difference is one of abbreviation
rather than invention. The work is real and we analyse its TAUM gate in detail in Sec. 3.2
and in our rebuttal, where we also reimplemented it from the authors' released code.
Correct entry: Zheng, Zhijie and Xiang, Xinhao and Zhang, Jiawei. "TTSA3R: Training-Free
Temporal-Spatial Adaptive Persistent State for Streaming 3D Reconstruction".
arXiv:2601.22615, 2026.

===============================================================================
APPEAL 5 — Bonn RGB-D Dynamic   [WEAK — use 其他 / Other]
===============================================================================
Reference identifier in paper:  \cite{bonn}  (dataset, Sec. 5 and App. A.2)

Exact citation as flagged:
@inproceedings{palazzolo2019bonn,
  author = {Emanuele Palazzolo and Jens Behley and Etienne Lozes and Philipp Gollub and Cyrill Stachniss},
  title = {Bonn RGB-D dynamic dataset for evaluation of 3D reconstruction approaches},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2019}
}

Authoritative sources:
Dataset artifact: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
  (officially named "Bonn RGB-D Dynamic Dataset")
Associated paper: https://dl.acm.org/doi/10.1109/IROS40897.2019.8967590
  "ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting
  Residuals", Palazzolo, Behley, Lottes, Giguère, Stachniss, IROS 2019.

Basis: Other.

Explanation:
We do not dispute that two author surnames are wrong ("Etienne Lozes" and "Philipp Gollub"
in place of "Philipp Lottes" and "Philippe Giguère"), and we correct them below. We appeal
the classification of the reference as pointing to a non-existent work. What we cite and
what we actually use is the dataset, which exists publicly under exactly the name we give
it, "Bonn RGB-D Dynamic Dataset", hosted by the University of Bonn at the link above. The
stated criteria note that references to existing artifacts that do not correspond to a
single document "are fine". Two of the five authors (Palazzolo, Behley) and Stachniss, the
venue and the year are correct.
Correct entry: Palazzolo, Emanuele and Behley, Jens and Lottes, Philipp and Giguère,
Philippe and Stachniss, Cyrill. "ReFusion: 3D Reconstruction in Dynamic Environments for
RGB-D Cameras Exploiting Residuals". IROS 2019.

===============================================================================
APPEAL 6 — CUT3R   [WEAKEST — use 其他 / Other]
===============================================================================
Reference identifier in paper:  \cite{cut3r}  (base model, throughout)

Exact citation as flagged:
@article{wang2025cut3r,
  author = {Jiawei Wang et al.},
  title = {Cut3r: Cutting-edge 3d reconstruction by learning to reconstruct the world from randomly generated videos},
  journal = {arXiv preprint arXiv:2504.00409},
  year = {2025}
}

Authoritative source for the cited work:
https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html
Secondary: https://arxiv.org/abs/2501.12387  |  https://cut3r.github.io/
"Continuous 3D Perception Model with Persistent State", Qianqian Wang, Yifei Zhang,
Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa. CVPR 2025.

Basis: Other.

Explanation:
We do not dispute the checker's finding on this entry. The subtitle text and the arXiv
identifier in our citation are wrong, and the first author's given name is wrong. We
appeal only the inference that this indicates a reference to a work that does not exist.
CUT3R is the base model of our submission. It is named in our title, abstract, method and
every results table, our entire codebase is built on the authors' released CUT3R
implementation and checkpoint, and every experiment in the paper runs on it. The acronym
"CUT3R" in the cited title is the model's own name and the standard way the community
refers to this work. The defect is a placeholder .bib entry whose metadata we failed to
verify before submission.
Correct entry: Wang, Qianqian and Zhang, Yifei and Holynski, Aleksander and Efros,
Alexei A. and Kanazawa, Angjoo. "Continuous 3D Perception Model with Persistent State".
CVPR 2025. arXiv:2501.12387.
