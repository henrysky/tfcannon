import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class TFcannon():
    def __init__(self, regularizer=None):
        """
        The Cannon implementation with Tensorflow

        :param regularizer: Regularization
        :type regularizer: float
        :return: None
        :History: 2019-Aug-02 - Written - Henry Leung (University of Toronto)
        """
        self.regularizer = regularizer
        self.nspec = None
        self.npixels = None
        self.nlabels = None
        self.ncoeffs = None

        self.coeffs = None
        self.scatter = None

    def train(self, spec, specerr, labels):
        """
        Training

        :param spec: spectra
        :type spec: ndarray
        :param specerr: spectra-err
        :type specerr: ndarray
        :return: None
        :History: 2019-Aug-02 - Written - Henry Leung (University of Toronto)
        """
        # just in case only one label is provided
        labels = np.atleast_2d(labels)

        self.nspec = spec.shape[0]
        self.npixels = spec.shape[1]
        self.nlabels = labels.shape[1]

        self.ncoeffs = (self.nlabels * (self.nlabels + 3)) // 2 + 1
        self.coeffs = np.zeros((self.ncoeffs, self.npixels)) + np.nan
        self.scatter = np.zeros(self.npixels) + np.nan

        tf_specs = tf.convert_to_tensor(spec, dtype=tf.float32)
        tf_specs_err = tf.convert_to_tensor(specerr, dtype=tf.float32)
        tf_labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # internal label matrix
        stack_labels = self._quad_terms(tf.concat([tf.ones([self.nspec, 1], dtype=tf.float32), tf_labels], axis=1))

        # initial
        init_scatter = tf.math.sqrt(tf.math.reduce_variance(tf_specs, axis=1) -
                                    tfp.stats.percentile(tf_specs_err, 50, axis=1) ** 2.)

        init_scatter = tf.where(tf.math.is_nan(init_scatter), tf.math.reduce_std(spec, axis=1), init_scatter)
        final_coeffs = tf.ones([self.ncoeffs, 1], dtype=tf.float32) * -9999.
        final_scatter = tf.ones([1], dtype=tf.float32) * -9999.

        i = tf.constant(0)
        out = tf.while_loop(lambda _specs, _specserr, _labels, _init_scatter, _i, _, __: tf.less(_i, self.npixels),
                            self._quad_fitting,
                            [tf_specs, tf_specs_err, stack_labels, init_scatter, i, final_coeffs, final_scatter],
                            shape_invariants=[tf_specs.get_shape(),
                                              tf_specs_err.get_shape(),
                                              stack_labels.get_shape(),
                                              init_scatter.get_shape(),
                                              i.get_shape(),
                                              tf.TensorShape([self.ncoeffs, None]),
                                              tf.TensorShape([None, ])])

        with tf.Session() as sess:
            _, _, _, _, _, coeffs, scatter = sess.run(out)

        # set model coeffs
        self.coeffs[:, :] = coeffs[:, 1:]
        self.scatter[:] = scatter[1:]

    def test(self, spec, specerr):
        """
        Takes spectra and return best fit labels (based on numpy so far, numpy seems faster)

        :param spec: spectra
        :type spec: ndarray
        :param specerr: spectra-err
        :type specerr: ndarray
        :return: None
        :History: 2019-Aug-06 - Written - Henry Leung (University of Toronto)
        """
        # just in case only one spectrum is provided
        spec = np.atleast_2d(spec)
        specerr = np.atleast_2d(specerr)

        # Setup output
        nspec = spec.shape[0]
        ncoeffs = self.coeffs.shape[0]

        nlabels = int((-3 + np.sqrt(9 + 8 * (ncoeffs - 1)))) // 2

        out = np.empty((nspec, nlabels))

        for ii in range(nspec):
            deno = specerr[ii] ** 2. + self.scatter ** 2.
            Y = (spec[ii] - self.coeffs[0]) / deno
            ATY = np.dot(self.coeffs[1:], Y)
            CiA = self.coeffs[1:].T * np.tile(1. / deno, (self.coeffs[1:].T.shape[1], 1)).T
            ATCiA = np.dot(self.coeffs[1:], CiA)
            ATCiAinv = np.linalg.inv(ATCiA)
            out[ii] = np.dot(ATCiAinv, ATY)[:nlabels]

        return out

    def _quad_terms(self, padded):
        """
        Stack all the quadratic terms in the label metrix

        :param padded: Ones padded labels (i.e. offset + labels)
        :return: stacked tensor of labels (i.e. offset + linear terms + quadratic terms)

        :History: 2019-Aug-02 - Written - Henry Leung (University of Toronto)
        """

        def loop_body(_padded, ii):
            _padded = tf.concat(
                [_padded, _padded[:, ii + 1:self.nlabels + 1] * tf.expand_dims(_padded[:, ii + 1], axis=-1)],
                axis=1)
            return _padded, tf.add(ii, 1)

        i = tf.constant(0)
        all_padded = tf.while_loop(lambda _padded, _i: tf.less(_i, self.nlabels),
                                   loop_body,
                                   [padded, i],
                                   shape_invariants=[tf.TensorShape([self.nspec, None]),
                                                     i.get_shape()])
        return all_padded[0]

    def _quad_fitting(self, specs, specs_err, labels, scatters, ii, final_coeffs, final_scatter):
        spec = specs[:, ii]
        spec_err = specs_err[:, ii]
        scatter = scatters[ii:ii + 1]

        def _quadfit_scatter_external(x):
            return self._quadfit_scatter(x, spec, spec_err, labels)

        fits = tfp.optimizer.lbfgs_minimize(_quadfit_scatter_external,
                                            scatter,
                                            x_tolerance=1e-7)
        result = fits.position

        final_coeffs = tf.concat(
            [final_coeffs, tf.expand_dims(self._polyfit_coeffs(spec, spec_err, result, labels), axis=1)], axis=1)
        final_scatter = tf.concat([final_scatter, result], axis=0)

        return specs, specs_err, labels, scatters, tf.add(ii, 1), final_coeffs, final_scatter

    def _quadfit_scatter(self, scatter, spec, spec_err, labelA):
        """
        Optimize the coefficients for this scatter
        """
        tcoeffs = self._polyfit_coeffs(spec, spec_err, scatter, labelA)
        # Get residuals for a given linear model of the spectra
        mspec = tf.math.reduce_sum(tcoeffs[1:self.nlabels + 1] * labelA[:, 1:self.nlabels + 1], axis=-1)

        def loop_body(_mspec, ii):
            label_terms = labelA[:, ii+1:ii+2] * labelA[:, ii + 1:self.nlabels + 1]
            terms = tf.math.reduce_sum((tcoeffs[self.nlabels + 1 + (ii * (2 * self.nlabels + 1 - ii)) // 2:
                                                self.nlabels + 1 + (ii * (
                                                            2 * self.nlabels + 1 - ii)) // 2 + self.nlabels - ii]) *
                                       label_terms,
                                       axis=1)
            _mspec = tf.add(_mspec, terms)
            return _mspec, tf.add(ii, 1)

        i = tf.constant(0)
        sum_mspec = tf.while_loop(lambda _padded, _i: tf.less(_i, self.nlabels),
                                  loop_body,
                                  [mspec, i],
                                  shape_invariants=[tf.TensorShape([self.nspec, ]),
                                                    i.get_shape()])[0]

        tres = spec - sum_mspec - tcoeffs[0]
        deno = spec_err ** 2. + scatter ** 2.
        output = 0.5 * tf.math.reduce_sum(tres ** 2. / deno) + 0.5 * tf.math.reduce_sum(tf.math.log(deno))

        return output, tf.gradients(output, scatter)[0]

    def _polyfit_coeffs(self, spec, specerr, scatter, labelA):
        """
        For a given scatter, return the best-fit coefficients by linear algebra
        """
        deno = specerr ** 2. + scatter ** 2.
        Y = spec / deno
        ATY = tf.tensordot(tf.transpose(labelA), Y, axes=[[1], [0]])
        C = tf.expand_dims(1. / deno, 0)
        CiA = labelA * tf.transpose(tf.tile(C, [tf.shape(labelA)[1], 1]))
        ATCiA = tf.tensordot(tf.transpose(labelA), CiA, axes=[[1], [0]])
        ATCiAinv = tf.linalg.inv(ATCiA)
        return tf.tensordot(ATCiAinv, ATY, axes=[[1], [0]])
