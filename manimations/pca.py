import copy
from typing import List

import numpy as np
from manim import (DL, DR, LEFT, MAROON, ORIGIN, PI, PURE_RED, UL, UR, YELLOW,
                   ApplyMethod, Circle, Create, Dot, Ellipse, FadeIn, FadeOut,
                   Group, Line, Matrix, NumberPlane, Scene, Tex, Text,
                   Transform, TransformFromCopy, Unwrite, Vector, VGroup,
                   Write, interpolate_color)

d = (np.sqrt(2) + np.sqrt(0.5) / 2) / 2
anti_d = (np.sqrt(2) - np.sqrt(0.5) / 2) / 2
TRANSFORM_M = np.array([[d, anti_d], [anti_d, d]])
OPACITY = 0.2


def tranform_dot(dot: Dot, center: Vector = ORIGIN) -> Dot:
    pos = dot.get_center().copy() - center
    pos[:2] = np.dot(pos[:2], TRANSFORM_M)
    return Dot(pos + center, color=dot.get_color(), fill_opacity=dot.get_fill_opacity())


def transform_circle(radius: float) -> Ellipse:
    w = 2 * 2 * np.sqrt((radius**2) / 2)
    h = 2 * 0.5 * np.sqrt((radius**2) / 2)
    return Ellipse(width=w, height=h)


def set_color(dot: Dot, data_max: np.ndarray) -> None:
    position = dot.get_center()
    alpha = position[:2] / data_max
    alpha = np.sqrt(np.sum(alpha**2))
    alpha / np.max(alpha)
    dot.set_color(interpolate_color(PURE_RED, YELLOW, alpha))


def dots_to_dots(
    dots_from: List[Dot], dots_to: List[Dot], run_time: float | None = None
) -> List[Transform]:
    if run_time is None:
        return [
            Transform(dot_from, dot_to) for dot_from, dot_to in zip(dots_from, dots_to)
        ]
    else:
        return [
            Transform(dot_from, dot_to, run_time=run_time)
            for dot_from, dot_to in zip(dots_from, dots_to)
        ]


class PCA(Scene):
    def construct(self) -> None:
        CUST_CENTER = ORIGIN
        numberplane = NumberPlane(x_range=(-13, 13, 1), y_range=(-8, 8, 1))
        numberplane.move_to(CUST_CENTER)
        data = np.load("data/normal.npy")
        transfo = np.around(TRANSFORM_M, 2)
        data_max = np.max(np.abs(data), axis=0) + 1
        # Add Z axis
        data = np.c_[data, np.zeros((data.shape[0], 1))]

        # DOTS
        __dots = []
        for line in data:
            dot = Dot(line)
            set_color(dot, data_max)
            __dots.append(dot)

        mvDot = Dot((0, 2, 0))
        set_color(mvDot, data_max)
        mvDot2 = Dot((2, 2, 0))
        set_color(mvDot2, data_max)
        mvDot3 = Dot((-2, 2, 0))
        set_color(mvDot3, data_max)
        __dots.append(mvDot)
        __dots.append(mvDot2)
        __dots.append(mvDot3)
        gDots = Group(*__dots)
        gDots.move_to(gDots.get_center() + CUST_CENTER)

        dots = copy.deepcopy(__dots)
        # Dots tranformed
        __transf_dots = [tranform_dot(dot, center=CUST_CENTER) for dot in __dots]

        # Dots low opacity
        __dots_op = copy.deepcopy(__dots)
        for dot in __dots_op:
            dot.set_opacity(OPACITY)
        __transf_dots_op = [tranform_dot(dot, center=CUST_CENTER) for dot in __dots_op]

        # VECTORS
        mvDotVector = Vector(mvDot)
        movedDotVector = Vector(tranform_dot(mvDot))
        mvDotVector2 = Vector(mvDot2)
        movedDotVector2 = Vector(tranform_dot(mvDot2))
        mvDotVector3 = Vector(mvDot3)
        movedDotVector3 = Vector(tranform_dot(mvDot3))

        # CIRCLES
        r = np.sqrt(8)
        shape_circle = Circle(radius=r)
        shape_ellipse = transform_circle(r)
        shape_ellipse.rotate(PI / 4)

        # LINES
        pc0 = Line(DL * 10, UR * 10, color=MAROON)
        pc1 = Line(DR * 10, UL * 10, color=MAROON)
        pc0_text = Text("PC0", font_size=24, color=MAROON)
        pc1_text = Text("PC1", font_size=24, color=MAROON)
        pc0_text.move_to((4, 3, 0)).rotate(PI / 4)
        pc1_text.move_to((-2, 3, 0)).rotate(PI / 4)

        # TEXTS
        D = Matrix(
            [
                [f"{__transf_dots[0].get_x():.2f}", f"{__transf_dots[0].get_y():.2f}"],
                ["\\vdots", "\\vdots"],
                [f"{__transf_dots[1].get_x():.2f}", f"{__transf_dots[1].get_y():.2f}"],
            ]
        )
        tex_D = Tex("Data =").next_to(D, direction=LEFT)
        D_group = VGroup(D, tex_D)
        D_group.move_to((-4.5, 2.5, 1))

        M = Matrix(
            [
                [f"{__dots[0].get_x():.2f}", f"{__dots[0].get_y():.2f}"],
                ["\\vdots", "\\vdots"],
                [f"{__dots[1].get_x():.2f}", f"{__dots[1].get_y():.2f}"],
            ]
        )
        tex_M = Tex("M =").next_to(M, direction=LEFT)
        M_group = VGroup(M, tex_M)
        M_group.move_to((4.5, 2.5, 1))

        Z = Matrix(transfo)
        tex_Z = Tex("Z =").next_to(Z, direction=LEFT)
        Z_group = VGroup(Z, tex_Z)
        Z_group.move_to((-4.5, -2.5, 1))

        tex_transfo = Tex("M", "Z")
        tex_transfo.next_to(tex_D)

        tex_cov = Tex(r"$cov(Data)$", r"$\approx$", r"$\lambda \cdot Z'Z$")
        tex_cov.move_to((-4.3, 2.5, 1))

        #############
        # ANIMATION #
        #############
        self.add(numberplane)

        dots = copy.deepcopy(__transf_dots)
        self.play(*[FadeIn(dot, run_time=2) for dot in dots], Write(D_group))
        self.wait(3)

        next_dots = copy.deepcopy(__dots)
        self.play(
            *dots_to_dots(dots, next_dots, run_time=3),
            Write(M_group),
            ApplyMethod(D_group.set_opacity, OPACITY),
        )
        self.wait(2)

        next_dots = copy.deepcopy(__dots)
        self.play(*dots_to_dots(dots, next_dots))

        self.play(Create(shape_circle), run_time=3)
        self.play(Write(Z_group))
        self.play(
            FadeOut(D, run_time=0.5),
            ApplyMethod(tex_D.set_opacity, 1),
            TransformFromCopy(M_group, tex_transfo[0]),
            TransformFromCopy(Z_group, tex_transfo[1]),
        )
        self.wait(2)

        next_dots = copy.deepcopy(__transf_dots)
        self.play(
            *dots_to_dots(dots, next_dots),
            Transform(shape_circle, shape_ellipse),
            run_time=5,
        )
        self.play(Unwrite(M_group), Unwrite(tex_D), Unwrite(tex_transfo), run_time=0.5)

        self.play(
            Unwrite(Z_group),
            run_time=0.5,
        )

        # Vector not PC
        next_dots = copy.deepcopy(__dots_op)
        next_dots[-3].set_opacity(1)
        self.play(FadeOut(shape_circle), run_time=0.5)
        self.play(*dots_to_dots(dots, next_dots))

        self.play(Create(mvDotVector))

        next_dots = copy.deepcopy(__transf_dots_op)
        next_dots[-3].set_opacity(1)
        self.play(
            *dots_to_dots(dots, next_dots),
            TransformFromCopy(mvDotVector, movedDotVector),
            ApplyMethod(mvDotVector.set_opacity, OPACITY),
            run_time=5,
        )

        # Vector PC0
        next_dots = copy.deepcopy(__dots_op)
        next_dots[-2].set_opacity(1)
        self.play(
            *dots_to_dots(dots, next_dots),
            FadeOut(*[mvDotVector, movedDotVector], run_time=0.5),
        )

        self.play(Create(mvDotVector2), run_time=2)

        next_dots = copy.deepcopy(__transf_dots_op)
        next_dots[-2].set_opacity(1)
        self.play(
            *dots_to_dots(dots, next_dots),
            TransformFromCopy(mvDotVector2, movedDotVector2),
            ApplyMethod(mvDotVector2.set_opacity, OPACITY),
            run_time=5,
        )

        self.play(Create(pc0), Write(pc0_text))

        # Vector pc1
        next_dots = copy.deepcopy(__dots_op)
        next_dots[-1].set_opacity(1)
        self.play(
            *dots_to_dots(dots, next_dots),
            FadeOut(*[mvDotVector2, movedDotVector2], run_time=0.5),
        )

        self.play(Create(mvDotVector3), run_time=2)

        next_dots = copy.deepcopy(__transf_dots_op)
        next_dots[-1].set_opacity(1)
        self.play(
            *dots_to_dots(dots, next_dots),
            TransformFromCopy(mvDotVector3, movedDotVector3),
            ApplyMethod(mvDotVector3.set_opacity, OPACITY),
            run_time=5,
        )

        self.play(Create(pc1), Write(pc1_text))

        # PCA rotation
        self.add_foreground_mobjects(*dots)
        next_dots = copy.deepcopy(__transf_dots)
        self.play(
            *dots_to_dots(dots, next_dots), FadeOut(*[mvDotVector3, movedDotVector3])
        )

        self.play(
            ApplyMethod(pc0.rotate, -PI / 4),
            ApplyMethod(pc1.rotate, -PI / 4),
            ApplyMethod(pc0_text.rotate, -PI / 4, {"about_point": ORIGIN}),
            ApplyMethod(pc1_text.rotate, -PI / 4, {"about_point": ORIGIN}),
            *[
                ApplyMethod(dot.rotate, -PI / 4, {"about_point": ORIGIN})
                for dot in dots
            ],
            run_time=4,
        )

        # PCA Projection
        self.play(
            *[
                ApplyMethod(dot.move_to, pc0.get_projection(dot.get_center()))
                for dot in dots
            ]
        )

        self.wait(5)
