import numpy as np
import scipy.stats as sps


class CollisionsGrid:
    def __init__(self, num_col=10000, v_cut=V_CUT):
        self.v_cut = v_cut
        self.num_col = num_col

        self.col_grid = self.set_col_grid()
        self.teta = 2 * np.arccos(self.col_grid[:, 6])
        self.ksi_update, self.ksi1_update = self.create_updated_velocities()

        assert np.all(((self.ksi_update ** 2).sum(axis=1) +
                       (self.ksi1_update ** 2).sum(axis=1) -
                       (self.col_grid[:, :6] ** 2).sum(axis=1)) < 10 ** (-9))

    def set_col_grid(self) -> np.ndarray:
        """
        function that create initial grid with 2 velocities, b and eps paameters
        :return: col_grid (num_col, 8)
        """
        grid = sps.uniform.rvs(size=(self.num_col, 8))
        grid[:, :6] = 2 * self.v_cut * grid[:, :6] - self.v_cut
        grid[:, 7] *= 2 * np.pi
        return grid

    def create_updated_velocities(self) -> (np.ndarray, np.ndarray):
        """
        Creating new velocities after collision
        :return: ksi_update, ksi1_update - two np.ndarray (num_col, 3)
        """
        g = self.col_grid[:, 3:6] - self.col_grid[:, :3]

        # g_1 = np.zeros_like(g)

        # Обновление массива g по формулам 1.12:
        indexes_0 = (g[:, 0] == 0) & (g[:, 1] == 0)
        indexes_1 = (g[:, 0] != 0) | (g[:, 1] != 0)
        g[indexes_0] = self.update_g_typeb(g[indexes_0], indexes_0)
        g[indexes_1] = self.update_g_typea(g[indexes_1], indexes_1)

        ksi_update = (self.col_grid[:, 3:6] + self.col_grid[:, :3] - g) / 2
        ksi1_update = (self.col_grid[:, 3:6] + self.col_grid[:, :3] + g) / 2

        return ksi_update, ksi1_update

    def update_g_typea(self, g: np.ndarray, indexes: np.ndarray) -> np.ndarray:
        """
        Updating relative velocities in regular case
        :param g: np.ndarray (sum(indexes), 3) - relative velocities
        :param indexes: np.ndarray - (num_col,) - bool matrix with regular cases
        :return: updated g - np.ndarray (sum(indexes), 3)
        """
        g_abs = np.sqrt((g ** 2).sum(axis=1))
        g_xy = np.sqrt((g[:, :2] ** 2).sum(axis=1))
        teta = self.teta[indexes]
        eps = self.col_grid[:, 7]

        g_1 = np.zeros_like(g)

        g_1[:, 0] = (g[:, 0] * np.cos(teta) - g[:, 0] * g[:, 2] * np.cos(eps) * np.sin(teta) / g_xy +
                     g_abs * g[:, 1] * np.sin(teta) * np.sin(eps) / g_xy)
        g_1[:, 1] = (g[:, 1] * np.cos(teta) - g[:, 1] * g[:, 2] * np.cos(eps) * np.sin(teta) / g_xy -
                     g_abs * g[:, 0] * np.sin(teta) * np.sin(eps) / g_xy)
        g_1[:, 2] = g[:, 2] * np.cos(teta) + g_xy * np.cos(eps) * np.sin(teta)

        return g_1

    def update_g_typeb(self, g, indexes):
        """
        Updating relative velocities in irregular case
        :param g: np.ndarray (sum(indexes), 3) - relative velocities
        :param indexes: np.ndarray - (num_col,) - bool matrix with irregular cases
        :return: updated g - np.ndarray (sum(indexes), 3)
        """
        g_abs = (g ** 2).sum(axis=1)
        teta = self.teta[indexes]

        g_1 = np.zeros_like(g)

        g_1[:, 0] = g_abs * np.sin(self.col_grid[indexes, 7]) * np.sin(teta)
        g_1[:, 1] = g_abs * np.cos(self.col_grid[indexes, 7]) * np.sin(teta)
        g_1[:, 2] = g_abs * np.cos(teta)
        return g_1