import numpy as np
from scipy.signal import fftconvolve

class VeinMatcher():
    def score(self, model, probe):
        """ Compute modified Miura score between the model and the probe

        In this modified version, no `ch` and `cw` parameters are used.
        The model should be instead cropped using it's mask before being scored.
        Original implementation: https://github.com/bioidiap/bob.bio.vein/blob/master/src/bob/bio/vein/algorithm/MiuraMatch.py

        Args:
            model (np.ndarray): The model image
            probe (np.ndarray): The probe image

        Returns:
            float: score between 0 and 0.5 (the higher the better)
        """
        model = model.astype(np.float64)
        probe = probe.astype(np.float64)
        mh, mw = model.shape

        Nm = fftconvolve(probe, np.rot90(model, k=2), "valid")
        t0, s0 = np.unravel_index(Nm.argmax(), Nm.shape)
        Nmm = Nm[t0, s0]

        score = Nmm / (model.sum() + probe[t0 : t0 + mh, s0 : s0 + mw].sum())
        return score