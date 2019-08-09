.. image:: https://travis-ci.org/henrysky/tfcannon.svg?branch=master
   :target: https://travis-ci.org/henrysky/tfcannon
   :alt: Build Status

Introduction
==============

``tfcannon`` is a version of ``the Cannon`` implemented with `Tensorflow`_ which enables it to be ran on NVIDIA GPU

If you find this package useful for your research, please cite original implementation paper `the Cannon`_ and `the Cannon 2`_
and acknowledge this repository as you like.

Installation
=================

``tfcannon`` requires ``python>=3.6`` and ``tensorflow>=1.14.0`` (including ``tensorflow>=2.0.0b1``) and ``tensorflow_probability>=0.7.0``

NVIDIA GPU is required if you want it to run on GPU

Just run ``python setup.py install`` to install or run ``python setup.py develop`` to develop

Tutorial
==========================

An example notebook with APOGEE DR14 is avaliable at here_

.. _here: tutorial/apogee_dr14_tutorial.ipynb


Basic Usage
============

.. code-block:: python

    from tfcannon import TFCannon

    model = TFCannon()

    # whether to force to use CPU even GPU presents
    model.force_cpu = False

    # x: your spectra
    # x_err: your spectra error
    # y: your labels
    # this will fit a quadratic relation
    model.train(x, x_err, y)

    # final coefficient and result
    model.coeffs, model.scatters

    # Best fit labels when you provide spectra
    best_fit_labels = model.test(spectra, spectra_err)

    # Generate spectra provided labels
    spectra = model.generate(labels)

    # save model
    model.save("cannon_model.h5")

    # load a saved model
    from tfcannon import load_model
    model = load_model("cannon_model.h5")

To do list
==========================

- Include function to find continuum
- Support censoring and regularization

Authors
=========
-  | **Henry Leung** - *Initial work and developer* - henrysky_
   | Student, Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - *Project Supervisor* - jobovy_
   | Professor, Department of Astronomy and Astrophysics, University of Toronto

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

License
---------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE
.. _galpy: https://github.com/jobovy/galpy
.. _Tensorflow: https://www.tensorflow.org/
.. _`the Cannon`: https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N/
.. _`the Cannon 2`: https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C/