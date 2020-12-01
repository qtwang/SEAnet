# coding = utf-8

from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as la
from tslearn.piecewise import PiecewiseAggregateApproximation
import scipy.fft
from pywt import wavedecn, waverecn

from util.pwlf import PiecewiseLinFit


class Summarization(ABC):
    def __init__(self, original: np.ndarray, summarized: np.ndarray = None, reconstructed: np.ndarray = None):
        assert original is not None
        self.original = np.array(original)

        self.summarized = summarized
        if self.summarized is not None:
            self.summarized = np.array(self.summarized)

        self.reconstructed = reconstructed
        if self.reconstructed is not None:
            self.reconstructed = np.array(self.reconstructed)


    @abstractmethod
    def summarize(self):
        raise NotImplementedError


    @abstractmethod
    def reconstruct(self):
        raise NotImplementedError


    def diff(self, dist: str = 'Euclidean', aggregate: bool = True):
        # original and reconstructed should be np.ndarray
        
        assert dist in {'Euclidean', 'RMS'}

        if self.reconstructed is None:
            if self.summarized is None:
                self.summarize()
            self.reconstruct()

        if dist == 'Euclidean':
            distances = la.norm(self.original - self.reconstructed, axis=1)
        elif dist == 'RMS':
            distances = np.sqrt(np.mean(np.square(self.original - self.reconstructed), axis=-1))

        if aggregate:
            return np.mean(distances)
        else:
            return distances



class DEA(Summarization):
    def __init__(self, original: np.ndarray, reconstructed: np.ndarray):
        super(DEA, self).__init__(original, reconstructed=reconstructed)


    def summarize(self):
        raise NotImplementedError


    def reconstruct(self):
        raise NotImplementedError


class DFT(Summarization):
    def __init__(self, original: np.ndarray, summarized_length: int, mode: str = 'best', num_components: int = 7):
        super(DFT, self).__init__(original)
        # / 2 for positive & negative frequency
        # / 2 for complex
        assert summarized_length % 2 == 0
        assert mode in {'best', 'first'}

        self.mode = mode

        # TODO only first several frequencies are perserved
        self.length_frequencies2perserve = int(summarized_length / 2)

        if mode == 'best':
            assert self.length_frequencies2perserve > num_components
            self.length_frequencies2perserve = num_components


    def summarize(self):
        self.summarized = np.zeros(self.original.shape, dtype=complex)

        if self.mode == 'first':
            for i, spectrum in enumerate(scipy.fft.fft(self.original, axis=-1)):
                # assuming spectrum[0] = 0 as z-normalization
                self.summarized[i][1: 1 + self.length_frequencies2perserve] = spectrum[1: 1 + self.length_frequencies2perserve]
                # self.summarized[i][-self.length_frequencies2perserve: ] = spectrum[-self.length_frequencies2perserve: ]
                # self.summarized[i][-self.length_frequencies2perserve: ] = np.flip(spectrum[1: 1 + self.length_frequencies2perserve])
        elif self.mode == 'best':
            for i, spectrum in enumerate(scipy.fft.fft(self.original, axis=-1)):
                for j in spectrum.argsort()[-self.length_frequencies2perserve: ]:
                    self.summarized[i][j] = spectrum[j]
        else:
            raise ValueError('not support {:} mode'.format(self.mode))


    def reconstruct(self):
        self.reconstructed = scipy.fft.ifft(self.summarized).real



class PAA(Summarization):
    def __init__(self, original: np.ndarray, summarized_length: int):
        super(PAA, self).__init__(original)

        self.original = original
        self.paa = PiecewiseAggregateApproximation(n_segments=summarized_length)


    def summarize(self):
        self.summarized = np.squeeze(self.paa.fit_transform(self.original))


    def reconstruct(self):
        self.reconstructed = np.squeeze(self.paa.inverse_transform(self.summarized))



class DCT(Summarization):
    def __init__(self, original: np.ndarray, summarized_length: int, dct_type: int = 2):
        super(DCT, self).__init__(original)
        assert 0 < dct_type < 5 

        self.type = dct_type
        self.summarized_length = summarized_length


    def summarize(self):
        self.summarized = np.zeros(self.original.shape)

        for i, spectrum in enumerate(scipy.fft.dct(self.original, self.type, axis=-1)):
            # TODO spectrum[0] = 0 as z-normalization?
            self.summarized[i][:  self.summarized_length] = spectrum[:  self.summarized_length]


    def reconstruct(self):
        self.reconstructed = scipy.fft.idct(self.summarized, self.type, axis=-1)



class DWT(Summarization):
    def __init__(self, original: np.ndarray, summarized_length: int, wavelet: str):
        super(DWT, self).__init__(original)
        assert wavelet in {'db2', 'haar'}

        self.wavelet = wavelet
        self.level = int(np.ceil(np.log2(self.original.shape[-1] / summarized_length)))
        print('{:d} --> {:d} via level {:d}'.format(self.original.shape[-1], summarized_length, self.level))


    def summarize(self):
        self.coeffs = wavedecn(self.original, self.wavelet, mode='smooth', level=self.level, axes=-1)
        self.summarized = self.coeffs[0]

        print('exact summarized_length = {:d}'.format(self.summarized.shape[-1]))


    def reconstruct(self):
        for i in range(1, self.level + 1):
            self.coeffs[-i] = {k: np.zeros_like(v) for k, v in self.coeffs[-i].items()}

        self.reconstructed = waverecn(self.coeffs, self.wavelet, mode='smooth', axes=-1)
        



class PXA(Summarization):
    def __init__(self, original: np.ndarray, degree: int, num_segments: int):
        super(PXA, self).__init__(original)
        assert 0 <= degree <= 2

        self.x = np.fromiter(range(1, self.original.shape[-1] + 1), np.int)
        self.degree = degree
        self.num_segments = num_segments


    def summarize(self):
        self.summarized = []

        for series in self.original:
            self.summarized.append(PiecewiseLinFit(self.x, series, degree=self.degree))
            self.summarized[-1].fit(self.num_segments)


    def reconstruct(self):
        self.reconstructed = np.zeros(self.original.shape)

        for i, local_pwlf in enumerate(self.summarized):
            self.reconstructed[i] = local_pwlf.predict(self.x)
