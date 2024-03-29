from tqdm import tqdm
import warnings
from packaging import version

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf2_flag = True if version.parse("2.0.0b0") <= version.parse(tf.__version__) else False

if tf2_flag:
    # disable eager execution to be compatible with tf2
    tf.compat.v1.disable_eager_execution()


class TFCannon:
    def __init__(self, l1_regularization=0.):
        """
        The Cannon implementation with Tensorflow

        :param l1_regularization: Regularization
        :type l1_regularization: float
        :return: None
        :History: 2019-Aug-02 - Written - Henry Leung (University of Toronto)
        """
        self.l1_regularization = l1_regularization
        self.nspec = None
        self.npixels = None
        self.nlabels = None
        self.ncoeffs = None

        self.coeffs = None
        self.scatter = None

        # normalization factor
        self.labels_median = 0.
        self.labels_std = 1.

        # labels names
        self.label_names = ['Teff', 'Logg', 'M_H', 'Alpha_M']

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
        h5f.create_dataset('l1_regularization', data=self.l1_regularization)
        h5f.create_dataset('label_names', data=np.array(self.label_names, dtype='S'))
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
        :param norm_flag: | whether to normalize label or not (with median and std) or
                          | 'cannon2' to subtract median and divided by 2 * (97.5 percentile - 2.5 percentile)
        :type norm_flag: Union([bool, str])
        :return: None
        :History: 2019-Aug-02 - Written - Henry Leung (University of Toronto)
        """
        # just in case only one label is provided
        labels = np.atleast_2d(labels)

        if norm_flag is True:
            self.labels_median = np.median(labels, axis=0)
            self.labels_std = np.std(labels, axis=0)
            labels = (labels - self.labels_median) / self.labels_std
        elif isinstance(norm_flag, str):
            if norm_flag.lower() == 'cannon2':
                if self.l1_regularization != 1e3:
                    warnings.warn(f"It is recommended to set L1 regularization factor same as Cannon 2 which is 1000 "
                                  f"(You are using {self.l1_regularization}) when using Cannon 2 like normalization.",
                                  UserWarning)
                self.labels_median = np.median(labels, axis=0)
                self.labels_std = 2 * (np.percentile(labels, 97.5, axis=0) - np.percentile(labels, 2.5, axis=0))
                labels = (labels - self.labels_median) / self.labels_std
            else:
                raise ValueError(f"Unknown norm_flag={norm_flag}")
        else:
            pass

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

        def _objective_func_external(x):
            return self._quadfit_objective(x, tf_spec, tf_spec_err, stack_labels)

        fits = tfp.optimizer.lbfgs_minimize(_objective_func_external,
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
            for i in tqdm(np.arange(self.npixels)):
                a = sess.run([final_coeffs, result], feed_dict={tf_spec: spec[:, i],
                                                                tf_spec_err: specerr[:, i],
                                                                tf_scatter: init_scatter[i:i + 1]})
                # set model coeffs
                self.coeffs[:, i] = a[0]
                self.scatter[:] = a[1]
        self.trained_flag = True

    def test(self, spec, specerr, tensorflow=False):
        """
        Takes spectra and return best fit labels (based on numpy so far, numpy seems faster)

        :param spec: spectra
        :type spec: ndarray
        :param specerr: spectra-err
        :type specerr: ndarray
        :param tensorflow: whether to use Tensorflow or NumPy
        :type tensorflow: bool
        :return: None
        :History: 2019-Aug-06 - Written - Henry Leung (University of Toronto)
        """
        self.check_train_flag()
        # just in case only one spectrum is provided
        spec = np.atleast_2d(spec) - self.coeffs[0]
        specerr = np.atleast_2d(specerr)

        # Setup output
        nspec = spec.shape[0]

        out = np.empty((nspec, self.nlabels))

        if not tensorflow:
            for ii in range(nspec):
                deno = specerr[ii] ** 2. + self.scatter ** 2.
                Y = spec[ii] / deno
                ATY = np.dot(self.coeffs[1:], Y)
                CiA = self.coeffs[1:].T * np.tile(1. / deno, (self.coeffs[1:].T.shape[1], 1)).T
                ATCiA = np.dot(self.coeffs[1:], CiA)
                ATCiAinv = np.linalg.inv(ATCiA)
                out[ii] = np.dot(ATCiAinv, ATY)[:self.nlabels]
        else:
            # prepare configuration
            kwargs = {}
            if self.force_cpu:
                kwargs['device_count'] = {'GPU': 0}
                print("Forcing tfcannon to use CPU")
            if self.log_device_placement:
                kwargs['log_device_placement'] = True

            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(**kwargs)) as sess:
                tf_spec = tf.compat.v1.placeholder(tf.float32, shape=[self.nspec])
                tf_spec_err = tf.compat.v1.placeholder(tf.float32, shape=[self.nspec])
                tf_scatter = tf.compat.v1.placeholder(tf.float32, shape=self.scatter.shape)
                tf_labelA = tf.compat.v1.placeholder(tf.float32, shape=self.coeffs[1:].T.shape)

                tf_func = self._polyfit_coeffs(tf_spec, tf_spec_err, tf_scatter, tf_labelA)

                for ii in range(nspec):
                    a = sess.run(tf_func, feed_dict={tf_spec: spec[ii],
                                                     tf_spec_err: specerr[ii],
                                                     tf_scatter: self.scatter,
                                                     tf_labelA: self.coeffs[1:].T})
                    out[ii] = a[:self.nlabels]

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
        labels = np.atleast_2d(labels)

        # normalize labels
        labels = (labels - self.labels_median) / self.labels_std

        # in case of only 1 label, then append offset
        labels = np.hstack([np.ones([labels.shape[0], 1]), labels])

        result = np.zeros([labels.shape[0], self.npixels])

        # append quadratic terms
        counter = 0
        for label in labels:
            label = np.atleast_2d(label)
            for ii in range(self.nlabels):
                label = np.hstack([label, label[:, ii + 1:self.nlabels + 1] * np.atleast_2d(label[:, ii + 1])])
            result[counter] = np.dot(label, self.coeffs)
            counter += 1

        return result

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
                                   back_prop=False)
        return all_padded[0]

    def _quadfit_objective(self, scatter, spec, spec_err, labelA):
        """
        Optimize the coefficients with this objective function
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
                                  back_prop=False)[0]

        tres = spec - sum_mspec - tcoeffs[0]
        deno = spec_err ** 2. + scatter ** 2.

        tcoeffs_nobias = tcoeffs[1:]

        output = 0.5 * tf.math.reduce_sum(tres * tres / deno) + 0.5 * tf.math.reduce_sum(
            tf.math.log(deno)) + self.l1_regularization * tf.math.reduce_sum(tf.math.abs(tcoeffs_nobias))

        # TODO: Not taking gradient properly
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
