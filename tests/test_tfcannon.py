import unittest


class TFCannonTestCase(unittest.TestCase):
    def test_main(self):
        from tfcannon import TFCannon, load_model
        import numpy as np
        import h5py

        try:
            h5f = h5py.File('tests/apogee_dr14_test.h5', 'r')
        except OSError:
            try:
                h5f = h5py.File('tfcannon/tests/apogee_dr14_test.h5', 'r')
            except OSError:
                h5f = h5py.File('apogee_dr14_test.h5', 'r')

        spec = np.array(h5f["spectra"])
        spec_err = np.array(h5f["spectra_error"])
        labels = np.array(h5f["teff_logg_feh_mgh"])

        model = TFCannon(l1_regularization=0.)
        model.train(spec, spec_err, labels)
        label_before = model.test(spec, spec_err)
        model.save('saved_model.h5')

        _model = load_model('saved_model.h5')
        label_after = _model.test(spec, spec_err, tensorflow=True)
        # assert it is deterministic and tensorflow working fine
        np.testing.assert_almost_equal(label_before, label_after, decimal=1)
        # assert label is accurate more or less
        self.assertEqual(np.sum(np.abs(label_after[:, 0]-labels[:, 0]) < 100) > 4000, True)


if __name__ == '__main__':
    unittest.main()
