import manim as M
import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def pdf(x: npt.NDArray, y: npt.NDArray, gauss: npt.NDArray) -> npt.NDArray:
    return norm.pdf(x, gauss[0, 0], gauss[0, 1]) * norm.pdf(y, gauss[1, 0], gauss[1, 1])


def ratios(
    data: npt.NDArray, g1: npt.NDArray, g2: npt.NDArray, pi1: float, pi2: float
) -> npt.NDArray:
    p1 = pi1 * pdf(data[:, 0], data[:, 1], g1)
    p2 = pi2 * pdf(data[:, 0], data[:, 1], g2)
    ratio = p1 / (p1 + p2)
    return np.stack((ratio, 1 - ratio), axis=1)


def update_gauss(data: npt.NDArray, ratio: npt.NDArray) -> npt.NDArray:
    m = (data * ratio).sum(axis=0) / ratio.sum()
    std = np.sqrt(((data - m) ** 2 * ratio).sum(axis=0) / ratio.sum())
    return np.stack((m, std), axis=1)


def step(
    data: npt.NDArray, g1: npt.NDArray, g2: npt.NDArray, pi1: float, pi2: float
) -> tuple[npt.NDArray, npt.NDArray, float, float]:
    ratio = ratios(data, g1, g2, pi1, pi2)
    pi1 = ratio[:, 0].mean()
    pi2 = ratio[:, 1].mean()
    g1 = update_gauss(data, ratio[:, 0:1])
    g2 = update_gauss(data, ratio[:, 1:2])
    return g1, g2, pi1, pi2


class Gaussian(M.VGroup):
    def __init__(self, gaussian: npt.NDArray, color: M.ManimColor = M.RED) -> None:
        super().__init__(color=color)
        nb_ellipses = 5
        self._elipses = []
        for i in range(nb_ellipses):
            opacity = (nb_ellipses - i) / nb_ellipses
            self._elipses.append(
                M.Ellipse(
                    width=gaussian[0, 1] * i ,
                    height=gaussian[1, 1] * i ,
                    color=self.color,
                    fill_opacity=opacity,
                )
            )
            self.add(self._elipses[-1])
        self.move_to((gaussian[0, 0], gaussian[1, 0], 0))


class KMeans(M.MovingCameraScene):
    def construct(self) -> None:
        rng = np.random.default_rng(4329)
        group1 = rng.normal((0, 1), (1, 4), size=(50, 2))
        group2 = rng.normal((0, -2), (4, 1), size=(50, 2))
        data = np.concatenate((group1, group2), axis=0)
        data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        g1, g2 = rng.uniform((-5, 0), (5, 5), size=(2, 2, 2))
        pi1, pi2 = 0.5, 0.5

        # Plan
        self.camera.frame.scale(2.2)
        numberplane = M.NumberPlane(x_range=(-16, 16, 1), y_range=(-9, 9, 1))
        dots = [M.Dot(dot) for dot in data]
        g_dots = M.VGroup(*dots)
        gauss1 = Gaussian(g1)
        gauss2 = Gaussian(g2, color=M.BLUE)
        g_dots.set_z_index(2)
        self.add(numberplane)
        self.play(M.FadeIn(g_dots))
        self.play(M.GrowFromCenter(gauss1),M.GrowFromCenter(gauss2))
        for i in range(10):
            prev_gauss1, prev_gauss2 = gauss1, gauss2
            g1, g2, pi1, pi2 = step(data, g1, g2, pi1, pi2)
            gauss1 = Gaussian(g1)
            gauss2 = Gaussian(g2, color=M.BLUE)
            self.play(
                M.FadeTransform(prev_gauss1, gauss1),
                M.FadeTransform(prev_gauss2, gauss2),
            )

