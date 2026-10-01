# hbsaemp

Hierarchical Bayes Small Area Estimation using Python, built on Bambi, PyMC and ArviZ.
A Python port of the R package [hbsaems](https://madsyair.github.io/hbsaems/).

Documentation: <https://madsyair.github.io/hbsaemp/>
(Bahasa Indonesia: <https://madsyair.github.io/hbsaemp/id/>)

## Install

Install hbsaemp straight from the GitHub repository. It needs Python 3.12 or later.

```bash
pip install "hbsaemp[bambi] @ git+https://github.com/madsyair/hbsaemp"   # Python API
pip install "hbsaemp[gui] @ git+https://github.com/madsyair/hbsaemp"     # adds the graphical interface
```

`[gui]` includes everything in `[bambi]`.

## Quick start

Fit a Fay-Herriot model to the bundled example data and estimate every area:

```python
from hbsaemp import (ModelConfig, check_convergence, create_model,
                     estimate_areas, load_dataset)

df = load_dataset("data_fhnorm")

config = ModelConfig(draws=2000, tune=1000, chains=4, target_accept=0.95)
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D", config=config)
model.fit()

print(check_convergence(model).summary())   # R-hat, ESS, divergences
print(estimate_areas(model).summary())      # per-area mean, RSE, MSE, HDI
```

The same model can be built three ways. Use the simplest one that can express yours:

```python
from hbsaemp import hbm_flex, hbm_gaussian

beginner = hbm_gaussian("y", ["x1", "x2"], df, sampling_var="D", area_var="group")
intermediate = hbm_flex("y", ["x1", "x2"], df, family="gaussian",
                        sampling_var="D", area_var="group")
advanced = create_model("y ~ x1 + x2", family="gaussian", data=df,
                        group="group", sampling_var="D")
```

`hbm_beta` and `hbm_binomial` do the same for the beta and binomial families.

Coming from R? Every hbsaems function keeps its name as an alias (`hbm`, `hbpc`, `hbcc`,
`hbmc`, `hbsae`, `update_hbm`). See
[Migrate from the R package](https://madsyair.github.io/hbsaemp/python/how-to/migrate-from-hbsaems.html).

## Graphical interface

With the `[gui]` extra installed, start the app from a terminal:

```bash
hbsaemp-app
```

or from Python:

```python
from hbsaemp import launch_app

launch_app()
```

## Documentation

The documentation has two tracks. Each is complete on its own.

- [Python API](https://madsyair.github.io/hbsaemp/python/index.html): write the analysis as code.
- [Graphical interface](https://madsyair.github.io/hbsaemp/gui/index.html): work through the analysis by clicking.

Each track has tutorials, how-to guides, reference and explanation. The
[API reference](https://madsyair.github.io/hbsaemp/python/reference/index.html) mirrors the
package layout.

## Status

Version 1.0.0 (beta) supports the gaussian (Fay-Herriot), beta and binomial families.
Version 2.0.0 is planned to add spatial random effects and a lognormal family.

## Authors

- **Achmad Syahrul Choir**, author and maintainer.
  ORCID [0000-0001-7088-0646](https://orcid.org/0000-0001-7088-0646)
- **M. Ihsan Silmi Kaffah**, author.
- **Rahmadika Kemala Salsabiela**, author.

## Citation

> Choir A, Kaffah MIS, Salsabiela RK (2026). _hbsaemp: Hierarchical Bayes Small Area
> Estimation using Python_. Python package version 1.0.0. <https://madsyair.github.io/hbsaemp/>

```bibtex
@Manual{,
  title = {hbsaemp: Hierarchical Bayes Small Area Estimation using Python},
  author = {Achmad Syahrul Choir and M. Ihsan Silmi Kaffah and Rahmadika Kemala Salsabiela},
  year = {2026},
  note = {Python package version 1.0.0},
  url = {https://madsyair.github.io/hbsaemp/},
}
```

If you used a different release, cite that version.

## License

GPL-3.0-or-later, the same terms as hbsaems. See [LICENSE](LICENSE).

## References

- Capretto, T. et al. (2022). Bambi: A simple interface for fitting Bayesian linear models
  in Python. _Journal of Statistical Software_, 103(15).
  <https://doi.org/10.18637/jss.v103.i15>
- Choir, A. S., Nurhayati, S. S., Zamzanah, S., & Oktalia Siregar, A. L. (2026).
  _hbsaems: Hierarchical Bayesian Area-Level Small Area Estimation Models_. R package
  version 1.1.0. <https://madsyair.github.io/hbsaems/>
- Fay, R. E. & Herriot, R. A. (1979). Estimates of income for small places: an application
  of James-Stein procedures to census data. _Journal of the American Statistical
  Association_, 74(366), 269-277.
- Rao, J. N. K. & Molina, I. (2015). _Small Area Estimation_ (2nd ed.). Wiley.
