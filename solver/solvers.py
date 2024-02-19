import numpy as np
import matplotlib.pyplot as plt
from constants import *
from tqdm import tqdm


class SimpleF:
    def __init__(self, v_cut=V_CUT, n_v=N_V):
        self.v_cut = v_cut
        self.v_1 = np.linspace(-self.v_cut, self.v_cut, n_v)

        self.v_grid = np.array(np.meshgrid(self.v_1, self.v_1, self.v_1)).T

        self.f = self.set_initial_data()

    def set_initial_data(self):
        net = np.ones((N_X, N_Y)).reshape(N_X, N_Y, 1, 1, 1)

        f_v_1 = np.exp(-((self.v_grid - U) ** 2).sum(axis=3) / (2 * T_1))
        f = net * f_v_1 / f_v_1.sum()

        f[N_X * CHIP_X // X_MAX:N_X * (CHIP_X + CHIP_DX) // X_MAX, :N_Y * CHIP_DY // Y_MAX, :, :, :] = 0  # Чип

        f_v_2 = np.exp(-((self.v_grid - U) ** 2).sum(axis=3) / (2 * T_2))
        f[N_X * CHIP_X // X_MAX - 1, :N_Y * CHIP_DY // Y_MAX, :, :, :] = f_v_2 / f_v_2.sum()
        f[N_X * (CHIP_X + CHIP_DX) // X_MAX, :N_Y * CHIP_DY // Y_MAX, :, :, :] = f_v_2 / f_v_2.sum()
        f[N_X * CHIP_X // X_MAX:N_X * (CHIP_X + CHIP_DX) // X_MAX, N_Y * CHIP_DY // Y_MAX, :, :,
        :] = f_v_2 / f_v_2.sum()

        return f

    def update_data(self, new_data):
        self.f = new_data

    def show_state(self, name="n"):
        x_grid = np.linspace(0, X_MAX, N_X)
        y_grid = np.linspace(0, Y_MAX, N_Y)
        if name == "T":
            n = self.f.sum(axis=(2, 3, 4))
            content = (self.f * (self.v_grid ** 2).sum(axis=3)).sum(axis=(2, 3, 4)) / (3 * n)
        else:
            content = self.f.sum(axis=(2, 3, 4))
        plt.contourf(x_grid, y_grid, content.T, extend='both')
        # print(content[0, 0], content[50, 25])
        return content

    def compute_n(self, x_start, x_stop, y_start, y_stop):
        return self.f.sum(axis=(2, 3, 4))[x_start:x_stop, y_start:y_stop]

    def compute_T(self, x_start, x_stop, y_start, y_stop):
        n = self.f.sum(axis=(2, 3, 4))[x_start:x_stop, y_start:y_stop]
        return ((self.f * (self.v_grid ** 2).sum(axis=3)).sum(axis=(2, 3, 4)) / 3)[x_start:x_stop, y_start:y_stop] / n

    def compute_q(self, x, y):
        pass

    def compute_v(self, x, y):
        pass


class SimpleIdealSolver:
    def __init__(self, tau=TAU):
        self.tau = tau
        self.f = SimpleF()

    def make_timestep(self, t):
        ihalf = int(N_V / 2)

        # k > 0
        delta_x_left = self.f.f[1:, :, ihalf:, :, :] - self.f.f[:-1, :, ihalf:, :, :]
        k_x_left = (self.f.v_1[ihalf:] * self.tau / H_X).reshape(1, 1, -1, 1, 1)
        if t % 2 == 0:
            delta_y_left = self.f.f[:, 1:, :, ihalf:, :] - self.f.f[:, :-1, :, ihalf:, :]
            k_y_left = (self.f.v_1[ihalf:] * self.tau / H_Y).reshape(1, 1, 1, -1, 1)

        # k < 0
        delta_x_right = self.f.f[1:, :, :ihalf, :, :] - self.f.f[:-1, :, :ihalf, :, :]
        k_x_right = (self.f.v_1[:ihalf] * self.tau / H_X).reshape(1, 1, -1, 1, 1)
        if t % 2 == 0:
            delta_y_right = self.f.f[:, 1:, :, :ihalf, :] - self.f.f[:, :-1, :, :ihalf, :]
            k_y_right = (self.f.v_1[:ihalf] * self.tau / H_Y).reshape(1, 1, 1, -1, 1)

        new_f = np.copy(self.f.f)

        new_f[1:, :, ihalf:, :, :] -= k_x_left * delta_x_left
        new_f[:-1, :, :ihalf, :, :] -= k_x_right * delta_x_right

        if t % 2 == 0:
            new_f[:, 1:, :, ihalf:, :] -= k_y_left * delta_y_left
            new_f[:, :-1, :, :ihalf, :] -= k_y_right * delta_y_right

        # Границы:
        # Чип:
        icxl, icxr = N_X * CHIP_X // X_MAX, N_X * (CHIP_X + CHIP_DX) // X_MAX
        icy = N_Y * CHIP_DY // Y_MAX
        new_f[icxl:icxr, :icy, :, :, :] = 0

        f_v_2 = np.exp(-(self.f.v_grid ** 2).sum(axis=3) / (2 * T_2))
        v_abs = np.abs(self.f.v_1)

        # Левая граница:
        nom_h = (v_abs[ihalf:].reshape(1, -1, 1, 1) * self.f.f[icxl - 1, :icy + 1, ihalf:, :, :]).sum(axis=1)
        denom_h = (v_abs[:ihalf].reshape(-1, 1, 1) * f_v_2[:ihalf, :, :]).sum(axis=0)
        new_f[icxl - 1, :icy + 1, :ihalf, :, :] = (f_v_2[:ihalf, :, :] * nom_h.reshape(-1, 1, N_V, N_V) / denom_h)
        # print(self.f.compute_T(icxl-1, icxl, 0, icy))

        # Правая граница:
        nom_h = (v_abs[:ihalf].reshape(1, -1, 1, 1) * self.f.f[icxr, :icy + 1, :ihalf, :, :]).sum(axis=1)
        # denom_h = (v_abs[ihalf:].reshape(-1, 1, 1)*f_v_2[ihalf:, :, :]).sum(axis=0)
        new_f[icxr, :icy + 1, ihalf:, :, :] = f_v_2[ihalf:, :, :] * nom_h.reshape(-1, 1, N_V, N_V) / denom_h
        # print(self.f.compute_T(icxr, icxr+1, 0, icy))

        # Крышка
        # print(self.f.f[icxl:icxr, icy, :, :ihalf, :].shape)
        nom_h = (v_abs[:ihalf].reshape(1, 1, -1, 1) * self.f.f[icxl - 1:icxr + 1, icy, :, :ihalf, :]).sum(axis=2)
        # print(nom_h.shape)
        # denom_h = (v_abs[ihalf:].reshape(1, -1, 1)*f_v_2[:, ihalf:, :]).sum(axis=1)
        new_f[icxl - 1:icxr + 1, icy, :, ihalf:, :] = f_v_2[:, ihalf:, :] * nom_h.reshape(-1, N_V, 1,
                                                                                          N_V) / denom_h.reshape(N_V, 1,
                                                                                                                 N_V)

        # Стенки:
        f_v_2 = np.exp(-(self.f.v_grid ** 2).sum(axis=3) / (2 * T_1))
        # Верхняя стенка:
        nom_h = (v_abs[ihalf:].reshape(1, 1, -1, 1) * self.f.f[:, N_Y - 1, :, ihalf:, :]).sum(axis=2)
        denom_h = (v_abs[:ihalf].reshape(1, -1, 1) * f_v_2[:, :ihalf, :]).sum(axis=1)
        new_f[:, N_Y - 1, :, :ihalf, :] = (
                    f_v_2[:, :ihalf, :] * nom_h.reshape(-1, N_V, 1, N_V) / denom_h.reshape(N_V, 1, N_V))

        # Нижняя стенка:
        nom_h = (v_abs[:ihalf].reshape(1, 1, -1, 1) * self.f.f[:, 0, :, :ihalf, :]).sum(axis=2)
        denom_h = (v_abs[ihalf:].reshape(1, -1, 1) * f_v_2[:, ihalf:, :]).sum(axis=1)
        new_f[:, 0, :, ihalf:, :] = f_v_2[:, ihalf:, :] * nom_h.reshape(-1, N_V, 1, N_V) / denom_h.reshape(N_V, 1, N_V)

        # Входы:
        f_v_2 = np.exp(-((self.f.v_grid - U) ** 2).sum(axis=3) / (2 * T_1))
        new_f[0, :, :, :, :] = f_v_2 / f_v_2.sum()
        new_f[N_X - 1, :, :, :, :] = f_v_2 / f_v_2.sum()

        self.f.update_data(new_f)

    def solve(self, nsteps=10, name="n"):
        for t in tqdm(range(nsteps)):
            self.make_timestep(t)
            self.f.show_state(name)