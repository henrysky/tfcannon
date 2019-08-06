
Introduction
==============

``tfcannon`` is a version of `the Cannon` implemented with `Tensorflow`_

If you find this package usage for your research, please cite original implementation paper `the Cannon`_ and `the Cannon 2`_
and acknowledge this repository as you like.

**work in progress nothing is working properly**

**Mostly run without error**

Installation
=================

Just run ``python setup.py install`` to install or run ``python setup.py install`` to develop

Developed with ``tensorflow==1.14.0``

To do list
==========================

- Get it to work
- Investigate why some operations are still running on CPU??
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