V_CUT = 4.8
N_V = 20
T_1 = 1
T_2 = 2
X_MAX = 25
Y_MAX = 20
CHIP_X = 10
CHIP_DY = 1
CHIP_DX = 5
U = 0.01
N_X = 100
N_Y = 100

H_X = X_MAX / N_X
H_Y = Y_MAX / N_Y

TAU = min(H_X, H_Y) / (2*V_CUT)