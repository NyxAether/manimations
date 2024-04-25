import manim as M
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

NumberType = int | float


def get_rotation(gaussian: npt.NDArray) -> tuple[float, npt.NDArray]:
    eign_v, rot_M = np.linalg.eig(gaussian[1:])
    rot_M = np.concatenate((rot_M, np.zeros((2, 1))), axis=1)
    rot_M = np.concatenate((rot_M, np.zeros((1, 3))), axis=0)
    rot_M[2, 2] = 1
    return Rotation.from_matrix(rot_M).as_euler("xyz")[2], eign_v


def pdf(x: npt.NDArray, y: npt.NDArray, gauss: npt.NDArray) -> npt.NDArray:
    return multivariate_normal(gauss[0], gauss[1:], allow_singular=True).pdf(
        np.dstack((x, y))
    )


def ratios(
    data: npt.NDArray, gaussians: list[npt.NDArray], pis: npt.NDArray
) -> npt.NDArray:
    probas = []
    for g, pi in zip(gaussians, pis):
        probas.append(pi * pdf(data[:, 0], data[:, 1], g))
    np_probas = np.stack(probas, axis=1)
    ratio = np_probas / np_probas.sum(axis=1, keepdims=True)
    return ratio


def exp_mean_cov(data: npt.NDArray, ratio: npt.NDArray) -> npt.NDArray:
    m = (data * ratio).sum(axis=0) / ratio.sum()
    cov = (data - m).T @ (ratio * (data - m)) / ratio.sum()
    return np.concatenate((m[np.newaxis, ...], cov), axis=0)


def random_multivariate_normal(
    rng: np.random.Generator,
    mean_range: tuple[NumberType, NumberType],
    x_var_range: tuple[NumberType, NumberType],
    y_var_range: tuple[NumberType, NumberType],
) -> npt.NDArray:
    mean = rng.uniform(mean_range[0], mean_range[1], size=(1, 2))
    x_var = rng.uniform(*x_var_range)
    y_var = rng.uniform(*y_var_range)
    max_cov = np.sqrt(np.abs(x_var * y_var))
    cov = rng.uniform(-max_cov, max_cov)
    cov_mat = np.array([[x_var, cov], [cov, y_var]])
    return np.concatenate((mean, cov_mat), axis=0)


def step(
    data: npt.NDArray, gaussians: list[npt.NDArray], pis: npt.NDArray
) -> tuple[list[npt.NDArray], npt.NDArray]:
    r = ratios(data, gaussians, pis)
    pis = r.mean(axis=0)
    gaussians = [
        exp_mean_cov(data, r[:, i][..., np.newaxis]) for i in range(r.shape[1])
    ]
    return gaussians, pis


class Dot2Color(M.VGroup):
    def __init__(
        self,
        coord: npt.NDArray,
        color1: M.ManimColor = M.RED,
        color2: M.ManimColor = M.BLUE,
        ratio: float = 0.5,
    ) -> None:
        super().__init__()
        DOT_SIZE = 0.1
        self.coord = coord
        self.color1 = color1
        self.color2 = color2
        self.ratio = ratio
        self.sector1 = M.AnnularSector(
            0, 2 * DOT_SIZE, M.PI * 2 * self.ratio, color=self.color1
        )
        self.sector2 = M.AnnularSector(
            0, 2 * DOT_SIZE, -M.PI * 2 * (1 - self.ratio), color=self.color2
        )
        self.sector1.add_updater(
            lambda m: m.move_to(self.coord + np.array((0, DOT_SIZE, 0)))
        )
        self.sector2.add_updater(
            lambda m: m.move_to(self.coord - np.array((0, DOT_SIZE, 0)))
        )
        self.add(self.sector1, self.sector2)


class Gaussian(M.VGroup):
    def __init__(self, gaussian: npt.NDArray, color: M.ManimColor = M.RED) -> None:
        super().__init__(color=color)
        GAUSSIAN_SCALE = 1.2
        nb_ellipses = 5
        self._elipses = []
        rotation, eign_v = get_rotation(gaussian)
        for i in range(nb_ellipses):
            opacity = (nb_ellipses - i) / nb_ellipses
            self._elipses.append(
                M.Ellipse(
                    width=np.sqrt(eign_v[0]) * i * GAUSSIAN_SCALE,
                    height=np.sqrt(eign_v[1]) * i * GAUSSIAN_SCALE,
                    color=self.color,
                    fill_opacity=opacity,
                )
            )
        self.add(*self._elipses)
        self.move_to((gaussian[0, 0], gaussian[0, 1], 0))
        self.rotate(rotation)


class KMeans(M.MovingCameraScene):
    def construct(self) -> None:
        GROUP_SIZE = 200
        rng = np.random.default_rng(4339)
        group1 = rng.multivariate_normal(
            (-1, 1), ((6, 4), (4, 6)), size=GROUP_SIZE
        )  # mean, cov
        group2 = rng.multivariate_normal((0, -1), ((5, -4), (-4, 5)), size=GROUP_SIZE)
        group3 = rng.multivariate_normal((-4, 6), ((1, 0), (0, 1)), size=GROUP_SIZE)
        groups = [group1, group2, group3]
        data = np.concatenate(groups, axis=0)
        groups_as_points = [
            np.concatenate((gr, np.zeros((gr.shape[0], 1))), axis=1) for gr in groups
        ]
        data_as_points = np.concatenate((data, np.zeros((data.shape[0], 1))), axis=1)
        g1 = random_multivariate_normal(rng, (-5, 5), (0, 10), (0, 10))
        g2 = random_multivariate_normal(rng, (-5, 5), (0, 10), (0, 10))
        g3 = random_multivariate_normal(rng, (-5, 5), (0, 10), (0, 10))
        gaussians = [g1, g2, g3]
        colors = [M.GREEN, M.BLUE, M.RED]
        pis = np.ones(len(gaussians)) / len(gaussians)

        # Plan
        self.camera.frame.scale(2.2)
        numberplane = M.NumberPlane(x_range=(-16, 16, 1), y_range=(-9, 9, 1))
        self.add(numberplane)

        # Dots
        groups_g = []
        for group, color in zip(groups_as_points, colors[::-1]):
            current_gr = M.VGroup(*[M.Dot(dot, color=color) for dot in group])
            groups_g.append(current_gr)
            self.play(M.GrowFromCenter(current_gr))
        self.wait(1)

        dots = [M.Dot(dot) for dot in data_as_points]
        g_dots = M.VGroup(*dots)
        g_dots.set_z_index(2)
        self.play(*[M.FadeOut(gr) for gr in groups_g], M.FadeIn(g_dots))
        self.wait(1)

        # Gaussians
        g_gaussians = [Gaussian(g, color=color) for g, color in zip(gaussians, colors)]
        self.play(*[M.GrowFromCenter(g) for g in g_gaussians])
        self.wait(1)

        # Gaussians iteration
        for i in range(10):
            r = ratios(data, gaussians, pis)
            pis = r.mean(axis=0)
            gaussians = [
                exp_mean_cov(data, r[:, i][..., np.newaxis]) for i in range(r.shape[1])
            ]
            gaussians, pis = step(data, gaussians, pis)
            prev_gaussians = g_gaussians
            g_gaussians = [
                Gaussian(g, color=color) for g, color in zip(gaussians, colors)
            ]
            self.play(
                *[
                    M.ReplacementTransform(pg, g)
                    for pg, g in zip(prev_gaussians, g_gaussians)
                ]
            )
