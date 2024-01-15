# Tiny TCN by Zanghieri *et al*. (UNDER REVIEW)

This repository implements the Temporal Convolutional Network (TCN) [[1]](#1) [[2]](#2) presented in a manuscript by M. Zanghieri *et al*. [[3]](#3) currently under review at the IEEE *Access* journal.
We provide this as supplementary documentation for the peer review now in progress.
This small project is a preview of the TCN implementation; upon acceptance of the paper, we will publish the whole curated dataset and code developed for the project.
For a **technical report** about an earlier stage of the same research project, please refer to F. Conti *et al*. [[4]](#4).



## Usage

To run this small project, clone this repository:
```
git clone git@github.com:pulp-bio/tcn_under_review.git
```
The files expose the TCN's implementation and PyTorch and the file ``tcn_table.txt`` already contains the generated TCN's structure table.

The requirements (see ``requirements.txt``) are the Python packages PyTorch 1.9.0 and torchinfo 1.8.0, quickly installable via the shell with
```
python -m pip install -r requirements.txt
```
The TCN is implemented in the module ``tcn.py``.
1. Run ``visualize_tcn_table.ipynb`` (or equivalently ``visualize_tcn_table.py``) to generate the TCN, its ``torchinfo.ModelStatistics``, and the printed table.
2. See the printed table in the standard output or in the file ``tcn_table.txt``.

The generated output file is identical to the one already available in the repository.



## Authors

The manuscript documented by this repository involves several authors.
The piece of work reported here was developed at the **Energy-Efficient Embedded Systems (EEES) Lab** of University of Bologna (Italy) by:
- [Marcello Zanghieri](https://scholar.google.com/citations?user=WnIqQj4AAAAJ&hl=en) (Conceptualization, Software, Analysis)
- [Prof. Francesco Conti](https://scholar.google.it/citations?user=A70PCXoAAAAJ&hl=en) (Supervision, Funding acquisition)
- [Prof. Luca Benini](https://scholar.google.com/citations?user=8riq3sYAAAAJ&hl=en) (Supervision, Funding acquisition)

Prof. Luca Benini is also with the ETH Zürich (Switzerland).



## Citation

```
@article{key ,
    author = {Zanghieri, M. and others},
    title = {{MANUSCRIPT UNDER REVIEW}},
    journal = {IEEE Access},
    volume = {-},
    year = {2024},
    number = {-},
    pages = {--}
}
```



## References

<a id="1">[1]</a>
C. Lea *et al*., "Temporal convolutional networks for action segmentation and detection," *CoRR*, vol. [abs/1611.05267](https://doi.org/10.48550/arXiv.1611.05267), 2016.

<a id="2">[2]</a>
S. Bai *et al*., "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," *CoRR*, vol. [abs/1803.01271 (https://doi.org/10.48550/arXiv.1803.01271), 2018.

<a id="3">[3]</a>
M. Zanghieri *et al*., MANUSCRIPT UNDER REVIEW, [*IEEE Access*](https://ieeeaccess.ieee.org/).

<a id="4">[4]</a>
F. Conti *et al*., "AI-powered collision avoidance safety system for industrial woodworking machinery," in *AI4DI – Applications*. River Publishers, 2021. DOI: [10.1201/9781003337232-17](https://www.doi.org/10.1201/9781003337232-17).



## License

All files are released under the LGPL-2.1 license (`LGPL-2.1`) (see `LICENSE`).
