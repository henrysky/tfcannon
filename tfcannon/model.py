from tqdm import tqdm

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class TFCannon:
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

        # normalization factor
        self.labels_median = 0.
        self.labels_std = 1.

        self.trained_flag = False
        self.force_cpu = False
        self.log_device_placement = False

    def check_train_flag(self):
        if not self.trained_flag and self.coeffs is not None and self.scatter is not None:
            raise AttributeError("This model has not been trained")
        else:
            pass

    def save(self, name=None):
        """
        Save the model

        :param name: Name of the file to be saved
        :type name: str
        :return: None
        :History: 2019-Aug-06 - Written - Henry Leung (University of Toronto)
        """
        self.check_train_flag()

        if name is None:
            name = 'cannon_model'

        h5f = h5py.File(f'{name}', 'w')
        h5f.create_dataset('ceoffs', data=self.coeffs)
        h5f.create_dataset('scatter', data=self.scatter)
        h5f.create_dataset('npixel', data=self.npixels)
        h5f.create_dataset('nlabels', data=self.nlabels)
        h5f.create_dataset('labels_median', data=self.labels_median)
        h5f.create_dataset('labels_std', data=self.labels_std)
        h5f.close()

    def train(self, spec, specerr, labels, norm_flag=True):
        """
        Training

        :param spec: spectra
        :type spec: ndarray
        :param specerr: spectra-err
        :type specerr: ndarray
        :param labels: labels
        :type labels: ndarray
        :param norm_flag: whether to normalize label or not (with median and std)
        :type norm_flag: bool
        :return: None
        :History: 2019-Aug-02 - Written - Henry Leung (University of Toronto)
        """
        # just in case only one label is provided
        labels = np.atleast_2d(labels)

        if norm_flag:
            self.labels_median = np.median(labels, axis=0)
            self.labels_std = np.std(labels, axis=0)
            labels = (labels - self.labels_median) / self.labels_std

        self.nspec = spec.shape[0]
        self.npixels = spec.shape[1]
        self.nlabels = labels.shape[1]

        self.ncoeffs = (self.nlabels * (self.nlabels + 3)) // 2 + 1
        self.coeffs = np.zeros((self.ncoeffs, self.npixels)) + np.nan
        self.scatter = np.zeros(self.npixels) + np.nan

        init_scatter = np.sqrt(np.var(spec, axis=1) - np.median(specerr, axis=1) ** 2.)
        init_scatter[np.isnan(init_scatter)] = np.std(spec, axis=1)[np.isnan(init_scatter)]

        tf_labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # internal label matrix
        stack_labels = self._quad_terms(tf.concat([tf.ones([self.nspec, 1], dtype=tf.float32), tf_labels], axis=1))

        tf_spec = tf.compat.v1.placeholder(tf.float32, shape=[self.nspec])
        tf_spec_err = tf.compat.v1.placeholder(tf.float32, shape=[self.nspec])
        tf_scatter = tf.compat.v1.placeholder(tf.float32, shape=[1])

        def _quadfit_scatter_external(x):
            return self._quadfit_scatter(x, tf_spec, tf_spec_err, stack_labels)

        fits = tfp.optimizer.lbfgs_minimize(_quadfit_scatter_external,
                                            tf_scatter,
                                            x_tolerance=1e-7)
        result = fits.position

        final_coeffs = self._polyfit_coeffs(tf_spec, tf_spec_err, result, stack_labels)

        # prepare configuration
        kwargs = {}
        if self.force_cpu:
            kwargs['device_count'] = {'GPU': 0}
            print("Forcing tfcannon to use CPU")
        if self.log_device_placement:
            kwargs['log_device_placement'] = True

        print("Start Training")

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(**kwargs)) as sess:
            writer = tf.compat.v1.summary.FileWriter('./output', sess.graph)
            for i in tqdm(np.arange(self.npixels)):
                if i == self.npixels - 1:
                    run_metadata = tf.compat.v1.RunMetadata()
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    a = sess.run([final_coeffs, result], feed_dict={tf_spec: spec[:, i],
                                                                    tf_spec_err: specerr[:, i],
                                                                    tf_scatter: init_scatter[i:i + 1]},
                                 options=run_options,
                                 run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'Pixel {i}')
                else:
                    a = sess.run([final_coeffs, result], feed_dict={tf_spec: spec[:, i],
                                                                    tf_spec_err: specerr[:, i],
                                                                    tf_scatter: init_scatter[i:i + 1]})
                # set model coeffs
                self.coeffs[:, i] = a[0]
                self.scatter[:] = a[1]
            writer.close()
        self.trained_flag = True

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
        self.check_train_flag()
        # just in case only one spectrum is provided
        spec = np.atleast_2d(spec)
        specerr = np.atleast_2d(specerr)

        # Setup output
        nspec = spec.shape[0]

        out = np.empty((nspec, self.nlabels))

        for ii in range(nspec):
            deno = specerr[ii] ** 2. + self.scatter ** 2.
            Y = (spec[ii] - self.coeffs[0]) / deno
            ATY = np.dot(self.coeffs[1:], Y)
            CiA = self.coeffs[1:].T * np.tile(1. / deno, (self.coeffs[1:].T.shape[1], 1)).T
            ATCiA = np.dot(self.coeffs[1:], CiA)
            ATCiAinv = np.linalg.inv(ATCiA)
            out[ii] = np.dot(ATCiAinv, ATY)[:self.nlabels]

        # denormalize labels
        denorm_out = (out * self.labels_std) + self.labels_median

        return denorm_out

    def generate(self, labels):
        """
        Generate spectra from given  labels (based on numpy so far, numpy seems faster)

        :param labels: labels
        :type labels: ndarray
        :return: None
        :History: 2019-Aug-06 - Written - Henry Leung (University of Toronto)
        """
        self.check_train_flag()

        # normalize labels
        labels = (labels - self.labels_median) / self.labels_std

        # in case of only 1 label, then append offset
        labels = np.hstack([np.ones([1, 1]), np.atleast_2d(labels)])

        # append quadratic terms
        for ii in range(self.nlabels):
            labels = np.hstack([labels, labels[:, ii + 1:self.nlabels + 1] * np.atleast_2d(labels[:, ii + 1])])

        return np.dot(labels, self.coeffs)

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
                                                     i.get_shape()],
                                   parallel_iterations=1,
                                   back_prop=False,
                                   swap_memory=False,
                                   return_same_structure=True)
        return all_padded[0]

    def _quadfit_scatter(self, scatter, spec, spec_err, labelA):
        """
        Optimize the coefficients for this scatter
        """
        tcoeffs = self._polyfit_coeffs(spec, spec_err, scatter, labelA)
        # Get residuals for a given linear model of the spectra
        mspec = tf.math.reduce_sum(tcoeffs[1:self.nlabels + 1] * labelA[:, 1:self.nlabels + 1], axis=-1)

        def loop_body(_mspec, ii):
            label_terms = labelA[:, ii + 1:ii + 2] * labelA[:, ii + 1:self.nlabels + 1]
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
                                                    i.get_shape()],
                                  parallel_iterations=self.nlabels,
                                  back_prop=False,
                                  swap_memory=False,
                                  return_same_structure=True)[0]

        tres = spec - sum_mspec - tcoeffs[0]
        deno = spec_err ** 2. + scatter ** 2.

        # just a workaround, need to fix gradient later
        output = 0.5 * tf.math.reduce_sum(tres * tres / deno) + 0.5 * tf.math.reduce_sum(tf.math.log(deno))

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
