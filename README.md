# Project prayas: Bayesian A/B Testing

*Prayas* is a Bayesian A/B Testing framework written in Python and
used within [Avira](http://avira.com>) to make business decisions
in many different areas. 
The [prayas website](https://avira.github.io/prayas) provides a general
introduction, example notebooks, and references to the underlying
methodology.

## Minimal installation

To install the *prayas* package with minimal requirements and effort
use `pip`:
```
wget https://raw.githubusercontent.com/Avira/prayas/master/requirements-minimal.txt
pip install -r requirements-minimal.txt
pip install git+https://github.com/Avira/prayas.git
```
You can excute these commands either in a `virtualenv` or a 
`conda` environment.


## Full installation

Clone project:
```
git clone https://github.com/Avira/prayas.git
```

If you are using the [Anaconda Distribution](https://www.anaconda.com/distribution/),
change into the project directory and setup the `conda` environment:
```
conda env create -f environment.yml
jupyter labextension install @pyviz/jupyterlab_pyviz
conda activate khoj
```

If you are using `virtualenv`, change into the project directory
and setup the environment:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter labextension install @pyviz/jupyterlab_pyviz
```

Note that both `environment.yml` and `requirements.txt` install also
Jupyter Lab, Sphinx, etc., i.e., everything that is needed to develop the
package and work with the notebooks. In case you only want to use
the plain package, use `environment-minimal.yml` or 
`requirements-minimal.txt`.

Install current version into the environment:
```
python setup.py install
```

To combine all these steps, we also provide a makefile; execute one 
of the following lines depending on your needs:
```
make conda
make conda minimal=1
make virtualenv
make virtualenv minimal=2
```

To work with the example notebooks, change into the notebooks directory
and start Jupyter Lab:
```
cd docs/notebooks
jupyter lab
```

## Feedback

In case of any bug, comments, feature requests, etc. please open an
[issue](https://github.com/Avira/prayas/issues) or a
[pull request](https://github.com/Avira/prayas/pulls).

## Contributors

All contributors are listed in [CONTRIBUTORS](CONTRIBUTORS).

## License

[MIT](https://choosealicense.com/licenses/mit/) License.
Copyright (c) 2019-2020 Avira Operations GmbH & Co. KG
