
Introduction
==============

``tfcannon`` is a version of ``the Cannon`` implemented with `Tensorflow`_ which enables it to be ran on NVIDIA GPU

If you find this package useful for your research, please cite original implementation paper `the Cannon`_ and `the Cannon 2`_
and acknowledge this repository as you like.

**work in progress nothing is working properly**

**Mostly run without error**

Installation
=================

``tfcannon`` requires ``python>=3.6`` and ``tensorflow>=1.14.0`` and ``tensorflow_probability>=0.7.0``

NVIDIA GPU is required if you want it to run on GPU

Just run ``python setup.py install`` to install or run ``python setup.py develop`` to develop

To do list
==========================

- Get it to work
- Investigate why some operations are still running on CPU??
- Investigate why ``tfcannon`` is so demanding on GPU bandwidth (thus slower on ``geir``)
- Add travis-CI, because tensorflow easily be broken
- Include function to find continuum
- Support censoring

Usage
=======

.. code-block:: python

    from tfcannon import TFcannon

    model = TFcannon()
    # x: your spectra
    # x_err: your spectra error
    # y: your labels
    # this will fit a quadratic relation
    coeffs, scatter = model.train(x, x_err, y)
    # final coefficient and result

    # model coefficient


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