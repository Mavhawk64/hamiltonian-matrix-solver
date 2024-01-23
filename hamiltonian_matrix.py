import sympy as sp
# import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Lagrangian():
    def __init__(self, L0, L1, L2) -> None:
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.L = self.L0 + self.L1 + self.L2

    def latex(self) -> str:
        return sp.latex(self.L0 + self.L1 + self.L2)

    def __str__(self) -> str:
        return str(self.L0 + self.L1 + self.L2)


# Define time
t = sp.symbols('t')

# Define the symbols
# Define the constants
m, g, c, a, b, M, t, w, r, mu = sp.symbols('m g c a b M t \\omega r \\mu')
# Define the position
phi = sp.symbols('\\phi')
pos = [r, phi]
# Define the velocity
vel = [sp.symbols("\\dot{" + str(i) + "}") for i in pos]
# Define the momentum
mom = [sp.symbols("p_" + str(i)) for i in pos]
# convert to sympy matrix
mom = sp.Matrix(mom)


# User-defined Lagrangian using provided symbols such that it is in the form L = [L_0, L_1, L_2]
# Lagrangian parts
# L0 = m*g*l*sp.cos(pos[0])  # Probably just the potential
# L1 = 0  # Linear vel terms --> vel^T * a
# L2 = 1/2*(4*m*a**2*vel[1]**2*(sp.sin(pos[1]))**2 + m*l**2*vel[0]**2-4*m*a*l*vel[1]*vel[0] *
#           sp.sin(pos[1])*sp.cos(pos[0])+1/2*M*a**2*vel[1]**2)  # Quadratic vel terms --> vel^T * T * vel
L0 = sp.Rational(1, 2)*mu*w*w*r*r
L1 = 0
L2 = sp.Rational(1, 2)*mu*(vel[0]**2+r**2*vel[1]**2)

L = Lagrangian(L0, L1, L2)

print(f"{bcolors.OKBLUE}Lagrangian:{bcolors.ENDC}\n")
print(L.latex())
print("\n")

T = sp.zeros(len(pos), len(pos))

for i in range(len(pos)):
    for j in range(len(pos)):
        # T[i, j] = L2.diff(vel[i]).diff(vel[j])
        T[i, j] = sp.diff(L2, vel[i], vel[j])

print(f"{bcolors.OKBLUE}T-matrix:{bcolors.ENDC}\n")
print(sp.latex(T))
print("\n")

a_vec = sp.zeros(len(pos), 1)
# L1 = vel^T * a
for i in range(len(pos)):
    a_vec[i, 0] = sp.diff(L1, vel[i])

print(f"{bcolors.OKBLUE}a-vector:{bcolors.ENDC}\n")
print(sp.latex(a_vec))
print("\n")


# H = 1/2(mom^T - a_vec^T) * T^{-1} * (mom - a_vec) - L0
H = sp.Rational(1, 2)*((mom - a_vec).T * T.inv() * (mom - a_vec))[0] - L0

print(f"{bcolors.OKBLUE}Hamiltonian:{bcolors.ENDC}\n")
print(sp.latex(H.simplify()))
print("\n")
