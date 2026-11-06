import numpy as np
from itertools import product

#########################
## One Qubit Operators ##
#########################
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

ops = [I, X, Y, Z]

################
## Parameters ##
################
n = 4     # Number of qubits
d = 2**n  # Space dimension
m = d**2 - 1    # Number of operators (max is d**2 - 1)
h = 10    # Number of times we measure each operator
N = m * h # Total measurements
verbose = False

# assert m is on the right range
if not (1 <= m <= d**2 - 1):
    raise ValueError("Invalid number of observables")

##########################
## Generate Pauli Basis ##
##########################
pauli_basis = []
for indices in product(range(4), repeat=n):
    # Exclude the identity operator I ⊗ ... ⊗ I
    if all(i == 0 for i in indices):
        continue

    P = ops[indices[0]]
    for i in indices[1:]:
        P = np.kron(P, ops[i])

    pauli_basis.append(P)

# There should be d^2 - 1 traceless Pauli strings
assert len(pauli_basis) == d**2 - 1

#######################
## Construct O, Q, T ##
#######################

# Select m linearly independent observables
O = pauli_basis[:m]

# Since Tr(O_i O_j) = d * delta_ij:
Q = O.copy()

# Remaining operators complete the traceless Hermitian basis
T = pauli_basis[m:]

###########################
## Verify the Conditions ##
###########################

# Tr(O_i Q_a) = d * delta_ia
for i in range(m):
    for a in range(m):
        value = np.trace(O[i] @ Q[a])
        expected = d if i == a else 0
        assert np.isclose(value, expected)

# Tr(O_i T_b) = 0
for i in range(m):
    for Tb in T:
        assert np.isclose(np.trace(O[i] @ Tb), 0)

if verbose: print("All dual-basis conditions verified.")

#################################
## Define theta, phi, and rho0 ##
#################################

seed = 69
rng = np.random.default_rng(seed) # Create seeded default rng

theta = np.array(rng.uniform(-1, 1, size=len(Q)), dtype=complex)
phi = np.array(rng.uniform(-1, 1, size=len(T)), dtype=complex)
rho0 = (1/2**n)*np.eye(2**n, 2**n, dtype=complex) # For the time being, rho0 is the identity normalized

# Compute rho given parameters
rho = rho0
for a in range(len(Q)):
    rho += (1/d) * theta[a] * Q[a]
for b in range(len(T)):
    rho += (1/d) * phi[b] * T[b]

if verbose:
    print(f"theta = {theta}")
    print(f"phi = {phi}")
    print(f"rho0 = \n{rho0}")
    print(f"rho = \n{rho}\n")


#############
## Compute ##
#############
eye = np.eye(2**n, 2**n) # Define identity to simplify code below
dataset = np.zeros([m, 2])
true_prob = np.zeros([m, 2])
est_prob = np.zeros([m, 2])
vhat = np.zeros(m)
v = np.zeros(m)
print("v(ρ), v̂(ρ)| true prob | est prob")
for i in range(m):
    v[i] = float(np.real(np.trace(rho @ O[i])))

    proj_p = (1/2)*(eye + O[i])
    proj_m = (1/2)*(eye - O[i])

    prob_p = float(np.real(np.trace(proj_p @ rho)))
    prob_m = float(np.real(np.trace(proj_m @ rho)))

    true_prob[i] = [prob_p, prob_m]

    samples = rng.choice([0, 1], size=h, p=true_prob[i])

    count = np.bincount(samples, minlength=2) # [+1#, -1#]
    dataset[i] = count
    # print(f"i = {i}, prob_p = {prob_p*100:.2f}, prob_m = {prob_m*100:.2f}, all +1 count = {count[0]}, all -1 count = {count[1]}")

    est_prob_p = dataset[i, 0]/h
    est_prob_m = dataset[i, 1]/h

    est_prob[i] = [est_prob_p, est_prob_m]

    vhat[i] = est_prob_p - est_prob_m

    print(f"{v[i]:.2f}, {vhat[i]:.2f}| {true_prob[i][0]:.2f}, {true_prob[i][1]:.2f}| {est_prob[i][0]:.2f}, {est_prob[i][1]:.2f}")