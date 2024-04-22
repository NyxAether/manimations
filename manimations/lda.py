import numpy as np
from manim import (BLUE, DOWN, LEFT, MAROON, MED_LARGE_BUFF, MED_SMALL_BUFF,
                   RED, SMALL_BUFF, UL, UP, UR, WHITE, ApplyMethod, Create,
                   Dot, FadeIn, FadeTransform, GrowFromCenter, Line,
                   ManimColor, MathTex, Matrix, NumberPlane, Scene, Tex,
                   Transform, TransformFromCopy, Unwrite, Vector, VGroup,
                   Write)

CM = {0: BLUE, 1: RED}
CM_ALT = {0: BLUE, 1: RED}
FONTSIZE = 40


def create_eigvectors(cov: np.ndarray, color: ManimColor = WHITE) -> VGroup:
    eigvals, eigvectors = np.linalg.eig(cov)
    vectors = []
    for i in range(len(eigvals)):
        vectors.append(
            Vector(
                direction=(*(eigvectors[:, i] * np.sqrt(np.abs(eigvals[i]))), 0),
                color=color,
            )
        )
        vectors.append(
            Vector(
                direction=(*(eigvectors[:, i] * -np.sqrt(np.abs(eigvals[i]))), 0),
                color=color,
            )
        )
    return VGroup(*vectors)


def pca(cov: np.ndarray) -> np.ndarray:
    eigvals, eigvectors = np.linalg.eig(cov)
    order = np.argsort(eigvals)
    return eigvectors[:, order][:, -1]


class LDA(Scene):
    def construct(self) -> None:
        data = np.load("data/lda.npy")
        data[:, :2] = data[:, :2] / 3
        data_total = data[:, :2]
        data0 = data[data[:, 2] == 0][:, :2]
        data1 = data[data[:, 2] == 1][:, :2]
        center0 = np.array((*np.mean(data0, axis=0), 0))
        center1 = np.array((*np.mean(data1, axis=0), 0))
        cov0 = np.cov(data0.T)
        cov1 = np.cov(data1.T)
        T = np.cov(data_total.T)
        W = (cov0 + cov1) / 2
        Winv = np.linalg.inv(W)
        B = T - W
        S = np.linalg.inv(W) @ B

        # Plan
        numberplane = NumberPlane(x_range=(-13, 13, 1), y_range=(-8, 8, 1))

        # Dots
        dots0 = [
            Dot(point=(coord[0], coord[1], 0), color=CM[coord[2]])
            for coord in data[data[:, 2] == 0]
        ]
        dots1 = [
            Dot(point=(coord[0], coord[1], 0), color=CM[coord[2]])
            for coord in data[data[:, 2] == 1]
        ]
        g_dots0 = VGroup(*dots0)
        g_dots1 = VGroup(*dots1)
        g_dots = VGroup(g_dots0, g_dots1)

        # EigenVectors
        g_vectors0 = create_eigvectors(cov=cov0, color=CM_ALT[0]).add_updater(
            lambda x, group=g_dots0: x.move_to(group.get_center_of_mass())
        )
        g_vectors1 = create_eigvectors(cov=cov1, color=CM_ALT[1]).add_updater(
            lambda x, group=g_dots1: x.move_to(group.get_center_of_mass())
        )
        g_vectors_W = create_eigvectors(cov=W)
        g_vectors_Winv = create_eigvectors(cov=Winv)
        g_vectors_B = create_eigvectors(cov=B)
        g_vectors_S = create_eigvectors(cov=S)
        g_vectors = create_eigvectors(cov=T)

        symb_W = MathTex("W").add_updater(
            lambda x, v=g_vectors_W: x.next_to(v.get_corner(UL), buff=MED_SMALL_BUFF)
        )
        symb_Winv = MathTex("W^{-1}").add_updater(
            lambda x, v=g_vectors_Winv: x.next_to(v.get_corner(UL), buff=MED_SMALL_BUFF)
        )
        symb_B = MathTex("B").add_updater(
            lambda x, v=g_vectors_B: x.next_to(v.get_corner(UR), buff=MED_SMALL_BUFF)
        )
        symb_S = MathTex("S").add_updater(
            lambda x, v=g_vectors_S: x.next_to(v.get_corner(UR), buff=MED_SMALL_BUFF)
        )
        symb_T = MathTex("cov(\\bullet)").add_updater(
            lambda x, v=g_vectors: x.next_to(v.get_corner(UL), buff=SMALL_BUFF)
        )

        # Texts
        formule = Tex(
            "Séparation",
            "$=$",
            "$\\frac{\\text{Variance entre groupe}}{\\text{Variance de groupe}}$",
            font_size=FONTSIZE,
        )
        formule.move_to((-4, -3, 0))
        formule_step1 = Tex(
            "$\\frac{\\text{Variance entre groupe}}{W}$", font_size=FONTSIZE
        ).next_to(formule[1])
        formule_step2 = Tex("$\\frac{B}{W}$", font_size=FONTSIZE).next_to(formule[1])
        formule_step3 = Tex("$S$", font_size=FONTSIZE).next_to(
            formule[1], direction=LEFT
        )
        formule_step4 = Tex("$W^{-1}B$", font_size=FONTSIZE).next_to(formule[1])

        formule_W = Tex("$W=$", "Variance de groupe", font_size=FONTSIZE)
        formule_W.move_to((4, -3, 0))
        formule_W_step1 = Tex(
            "(cov({{$\\bullet$}})+cov({{$\\bullet$}}))/2", font_size=FONTSIZE
        )
        formule_W_step1.next_to(formule_W[0])
        formule_W_step1[1].set_color(CM[0])
        formule_W_step1[3].set_color(CM[1])

        formule_B = Tex("$B=$", "Variance entre groupe", font_size=FONTSIZE)
        formule_B.move_to((4, -3, 0))
        formule_B_step1 = Tex("cov({{$\\bullet$}})-W", font_size=FONTSIZE)
        formule_B_step1.next_to(formule_B[0])

        # Matrices
        Mcov = Matrix(np.around(T, 2))
        text_cov = Tex("cov({{$\\bullet$}})=", font_size=FONTSIZE).next_to(Mcov, LEFT)
        g_Mcov = VGroup(Mcov, text_cov)
        g_Mcov.move_to((4, 3, 0))

        Mcov0 = Matrix(np.around(cov0, 2))
        text_cov0 = Tex("cov({{$\\bullet$}})=", font_size=FONTSIZE).next_to(Mcov0, LEFT)
        text_cov0.set_color_by_tex("bullet", CM[0])
        g_Mcov0 = VGroup(Mcov0, text_cov0)
        g_Mcov0.move_to((4, 3, 0))

        Mcov1 = Matrix(np.around(cov1, 2))
        text_cov1 = Tex("cov({{$\\bullet$}})=", font_size=FONTSIZE).next_to(Mcov1, LEFT)
        text_cov1.set_color_by_tex("bullet", CM[1])
        g_Mcov1 = VGroup(Mcov1, text_cov1)
        g_Mcov1.next_to(g_Mcov0, DOWN)

        McovW = Matrix(np.around(W, 2)).next_to(formule_W[0])

        McovB = Matrix(np.around(B, 2)).next_to(formule_B[0])

        McovS = Matrix(np.around(S, 2)).next_to(formule[1])

        text_Winv = MathTex("W^{-1}=", font_size=FONTSIZE)
        McovWinv = Matrix(np.around(Winv, 2)).next_to(text_Winv[0])
        formule_Win = VGroup(McovWinv, text_Winv)

        #############
        # ANIMATION #
        #############

        # Init
        self.add(numberplane)
        self.play(FadeIn(g_dots0), *(FadeIn(dot) for dot in dots1))

        # Formule Texte
        self.play(Write(formule))

        # Explication W
        self.play(Write(formule_W))
        self.play(FadeTransform(formule[2], formule_step1))

        self.play(
            ApplyMethod(g_dots0.shift, -center0), ApplyMethod(g_dots1.set_opacity, 0.1)
        )
        self.play(Write(g_Mcov0))
        self.play(GrowFromCenter(g_vectors0), ApplyMethod(g_dots0.set_opacity, 0.5))
        self.play(
            ApplyMethod(g_dots0.shift, center0), ApplyMethod(g_dots1.set_opacity, 1)
        )

        self.play(
            ApplyMethod(g_dots1.shift, -center1),
            ApplyMethod(g_dots0.set_opacity, 0.1),
            ApplyMethod(g_vectors0.set_opacity, 0.1),
        )
        self.play(Write(g_Mcov1))
        self.play(GrowFromCenter(g_vectors1), ApplyMethod(g_dots1.set_opacity, 0.5))
        self.play(
            ApplyMethod(g_dots1.shift, center1),
            ApplyMethod(g_dots0.set_opacity, 0.5),
            ApplyMethod(g_vectors0.set_opacity, 1),
        )

        self.play(Transform(formule_W[1], formule_W_step1))
        self.wait(3)

        g_vectors0.clear_updaters()
        g_vectors1.clear_updaters()
        self.play(
            Transform(formule_W[1], McovW),
            Write(symb_W),
            Transform(g_vectors0, g_vectors_W),
            Transform(g_vectors1, g_vectors_W),
            run_time=2,
        )

        self.wait(3)

        self.play(
            Unwrite(g_Mcov0),
            Unwrite(g_Mcov1),
            ApplyMethod(formule_W.move_to, (-5, 2.5, 0)),
        )

        self.remove(g_vectors0)
        self.remove(g_vectors1)
        g_vectors_W_full = VGroup(symb_W, g_vectors_W)
        self.play(
            FadeTransform(g_vectors_W_full, formule_W[1]),
            ApplyMethod(g_dots0.set_opacity, 1),
            ApplyMethod(g_dots1.set_opacity, 1),
        )

        # Explication B

        self.play(Write(formule_B))
        self.wait(1)
        self.play(FadeTransform(formule_step1, formule_step2))

        self.wait(3)

        self.play(Transform(formule_B[1], formule_B_step1))

        self.wait(1)

        g_Mcov.align_to(formule_W, UP)
        self.play(Write(g_Mcov))
        self.wait(1)
        self.bring_to_front(g_vectors)
        self.bring_to_back(g_dots)
        self.play(GrowFromCenter(g_vectors), Write(symb_T))

        self.wait(3)

        g_vectors_W_full.next_to(formule_W, DOWN)
        self.play(TransformFromCopy(formule_W, g_vectors_W_full))
        self.wait(1)

        self.play(
            Transform(formule_B[1], McovB),
            Unwrite(symb_T),
            Write(symb_B),
            Transform(g_vectors, g_vectors_B),
            FadeTransform(g_vectors_W_full, g_vectors_B),
        )
        self.remove(g_vectors)

        self.wait(3)

        g_vectors_B_full = VGroup(g_vectors_B, symb_B)
        self.play(Unwrite(g_Mcov))
        self.play(ApplyMethod(formule_B.align_to, formule_W, UP))
        self.play(FadeTransform(g_vectors_B_full, formule_B[1]))

        self.wait(3)

        # Explanation S
        self.play(Transform(formule[0], formule_step3))
        self.wait(1)
        self.play(Transform(formule_step2, formule_step4))
        self.wait(2)

        g_vectors_W_full.next_to(formule_W, DOWN, buff=MED_LARGE_BUFF)
        g_vectors_B_full.next_to(formule_B, DOWN, buff=MED_LARGE_BUFF)
        self.play(
            TransformFromCopy(formule_B, g_vectors_B_full),
            TransformFromCopy(formule_W, g_vectors_W_full),
        )

        self.wait(2)

        formule_Win.move_to(formule_W).align_to(formule_W, LEFT)
        g_vectors_Winv.move_to(g_vectors_W)
        self.play(
            Transform(formule_W, formule_Win),
            Transform(g_vectors_W, g_vectors_Winv),
            Transform(symb_W, symb_Winv),
        )

        self.wait(3)

        self.remove(formule_step2)
        self.play(
            Transform(formule_step4, McovS),
            Transform(g_vectors_B, g_vectors_S),
            Transform(g_vectors_W, g_vectors_S),
            Write(symb_S),
            Unwrite(symb_B, run_time=0.25),
            Unwrite(symb_W),
        )

        self.wait(3)

        self.play(
            Unwrite(g_vectors_B),
            Unwrite(formule[:2]),
            Unwrite(formule_step4),
            Unwrite(formule_W),
            Unwrite(formule_B),
        )

        # PCA Projection

        v_pc0 = -pca(S)
        angle = np.arctan2(v_pc0[1], v_pc0[0])
        pc0 = Line((*(v_pc0 * 20), 0), (*(-v_pc0 * 20), 0), color=MAROON)
        self.play(Create(pc0))
        self.wait(1)
        self.play(
            Unwrite(g_vectors_W),
            Unwrite(symb_S),
        )
        dots_pca = VGroup(pc0, g_dots)

        self.wait(1)
        self.play(ApplyMethod(dots_pca.rotate, -angle))

        self.wait(1)
        self.play(
            *[
                ApplyMethod(dot.move_to, pc0.get_projection(dot.get_center()))
                for dots in g_dots
                for dot in dots
            ]
        )

        self.wait(5)
