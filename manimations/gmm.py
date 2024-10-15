import numpy as np
import numpy.typing as npt
from manim import (BLUE, GREEN, PI, RED, AnnularSector, Dot, Ellipse, FadeIn,
                   FadeOut, GrowFromCenter, ManimColor, NumberPlane,
                   ReplacementTransform, VGroup, MovingCameraScene,Group,Scene,Line, Create)
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


class Dot2Color(VGroup):
    def __init__(
        self,
        coord: npt.NDArray,
        color1: ManimColor = RED,
        color2: ManimColor = BLUE,
        ratio: float = 0.5,
    ) -> None:
        super().__init__()
        DOT_SIZE = 0.1
        self.coord = coord
        self.color1 = color1
        self.color2 = color2
        self.ratio = ratio
        self.sector1 = AnnularSector(
            0, 2 * DOT_SIZE, PI * 2 * self.ratio, color=self.color1
        )
        self.sector2 = AnnularSector(
            0, 2 * DOT_SIZE, -PI * 2 * (1 - self.ratio), color=self.color2
        )
        self.sector1.add_updater(
            lambda m: m.move_to(self.coord + np.array((0, DOT_SIZE, 0)))
        )
        self.sector2.add_updater(
            lambda m: m.move_to(self.coord - np.array((0, DOT_SIZE, 0)))
        )
        self.add(self.sector1, self.sector2)


class Gaussian(VGroup):
    def __init__(self, gaussian: npt.NDArray, color: ManimColor = RED) -> None:
        super().__init__(color=color)
        nb_ellipses = 6
        self._elipses = []
        rotation, eign_v = get_rotation(gaussian)
        # ---Weird code here---#
        # Sometimes rotation may change by +-pi/2 between two iterations
        # and eigen values are 'inverted'
        # This is due to the computation of eigen vectors.
        # It's mathematically equivalent but it affect visualization
        # creating a 'wobbling' effect since it invert width and height
        # of the ellipses
        # Code below fixes this
        if eign_v[0] < eign_v[1]:
            eign_v = eign_v[::-1]
            rotation = rotation + np.pi / 2
        # ---Weird code end---#
        for i in range(nb_ellipses):
            opacity = (nb_ellipses - i) / nb_ellipses
            self._elipses.append(
                Ellipse(
                    width=np.sqrt(eign_v[0]) * i,
                    height=np.sqrt(eign_v[1]) * i,
                    color=self.color,
                    fill_opacity=opacity,
                )
            )
        self.add(*self._elipses)
        self.move_to((gaussian[0, 0], gaussian[0, 1], 0))
        self.rotate(rotation)


class GMM(MovingCameraScene):
    def construct(self) -> None:
        GROUP_SIZE = 200
        NB_GAUSSIANS = 3
        rng = np.random.default_rng(4329)
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
        gaussians = [
            random_multivariate_normal(rng, (-5, 5), (0, 10), (0, 10))
            for _ in range(NB_GAUSSIANS)
        ]
        colors = [GREEN, BLUE, RED]
        pis = np.ones(len(gaussians)) / len(gaussians)

        # Plan
        self.camera.scale(2.2)
        numberplane = NumberPlane(x_range=(-16, 16, 1), y_range=(-9, 9, 1))
        self.add(numberplane)

        # Dots
        groups_g = []
        for group, color in zip(groups_as_points, colors[::-1]):
            current_gr = VGroup(*[Dot(dot, color=color) for dot in group])
            groups_g.append(current_gr)
            self.play(GrowFromCenter(current_gr))
        self.wait(1)

        dots = [Dot(dot) for dot in data_as_points]
        g_dots = VGroup(*dots)
        self.play(*[FadeOut(gr) for gr in groups_g], FadeIn(g_dots))
        self.wait(1)

        # Gaussians
        g_gaussians = [Gaussian(g, color=color) for g, color in zip(gaussians, colors)]
        g_dots_prev = g_dots
        dots = [Dot(dot) for dot in data_as_points]
        g_dots = VGroup(*dots)
        self.play(*[GrowFromCenter(g) for g in g_gaussians] ,FadeOut(g_dots_prev), FadeIn(g_dots))
        self.wait(1)

        # self.interactive_embed()
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
                    ReplacementTransform(pg, g)
                    for pg, g in zip(prev_gaussians, g_gaussians)
                ]
            )
