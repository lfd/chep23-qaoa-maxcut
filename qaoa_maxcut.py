import os
import json
import networkx as nx
import numpy as np
from time import time
import csv
import re
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from scipy.optimize import minimize
from itertools import product
from multiprocessing import Pool
#import logging
#logging.basicConfig(level=logging.INFO)

from qat.opt.max_cut import MaxCut
from qat.qpus.hook_linalg import LinAlg
from qat.qpus.hook_mps import MPS
from qat.plugins import ScipyMinimizePlugin, QuameleonPlugin, Nnizer, NISQCompiler
from qat.core.console import display
from qat.opt.circuit_generator import CircuitGenerator
from qat.core import HardwareSpecs, Topology, Batch
from qat.core.gate_set import GateSignature, GateSet
from qat.noisy import NoisyQProc
from qat.hardware import make_depolarizing_hardware_model
from qat.hardware.default import HardwareModel, DefaultGatesSpecification
from qat.quops import make_depolarizing_channel, QuantumChannelKraus

from qiskit.providers.fake_provider import FakeMontreal

from arcs2022.src.maxcut import generate_graph_from_density

def create_graph_from_edges(definition_file, num_nodes):
    print(f'Loading graph definition from {definition_file}')

    with open(definition_file, 'r') as f:
        edges = json.load(f)

    G = nx.Graph()
    G.add_nodes_from(np.arange(num_nodes))
    G.add_edges_from(edges)

    return G

def create_new_graph(definition_file, num_nodes, density):
    print(f'Creating new graph definition in {definition_file}')

    G = generate_graph_from_density(num_nodes, density)

    with open(definition_file, 'w') as f:
        f.write(json.dumps([[int(u),int(v)] for (u,v) in G.edges()]))

    return G

def log_result(params, expectation, elapsed_time, num_nodes, density, p, i, simulator, single_qubit_noise_prob, two_qubit_noise_prob, sim_method):
    result_dir = f'experiments/nodes{num_nodes}/density{density}/problem{i:02}/'
    result_dir_json = f'{result_dir}Results/{simulator}/{sim_method}/{single_qubit_noise_prob}/{two_qubit_noise_prob}/p{p:02}/'

    if os.path.exists(result_dir_json):
        print(f'Overwriting results in {result_dir_json}')
        [os.remove(result_dir_json + f) for f in os.listdir(result_dir_json)]
    else:
        print(f'Creating new result directory in {result_dir_json}')
        os.makedirs(result_dir_json)


    with open(result_dir_json + 'parameters.json', 'w') as f:
        f.write(json.dumps(params))

    with open(result_dir_json + 'expectation.json', 'w') as f:
        f.write(json.dumps(expectation))

    with open(result_dir_json + 'elapsed_time.json', 'w') as f:
        f.write(json.dumps(elapsed_time))

    # store as csv
    if not os.path.exists(result_dir + 'results.csv'):
        with open(result_dir + 'results.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['simulator', 'sim_method', 'single_qubit_noise_prob', 'two_qubit_noise_prob', 'num_qubits', 'graph_density', 'p', 'problem_idx', 'expectation', 'elapsed_time'])

    with open(result_dir + 'results.csv') as f:
        with open(result_dir + 'results.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([simulator, sim_method, single_qubit_noise_prob, two_qubit_noise_prob, num_nodes, density, p, i, expectation, elapsed_time])

def json_to_csv(result_dir):
    nums = re.findall(r'\d+[.]?\d*', result_dir)
    num_nodes, density, problem_idx, p = nums
    simulator = result_dir.split('/')[-3]
    with open(result_dir + 'optimization_trace.json', 'r') as f:
        last_expectation = json.loads(json.loads(f.read()))[-1]

    with open(result_dir + 'elapsed_time.json', 'r') as f:
        elapsed_time = json.loads(f.read())

    # store as csv
    result_dir = '/'.join(result_dir.split('/')[:-3]) + '/'
    if not os.path.exists(result_dir + 'results.csv'):
        with open(result_dir + 'results.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['simulator', 'num_qubits', 'graph_density', 'p', 'problem_idx', 'last_expectation', 'elapsed_time'])

    with open(result_dir + 'results.csv') as f:
        with open(result_dir + 'results.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([simulator, int(num_nodes), float(density), int(p), int(problem_idx), last_expectation, elapsed_time])

def load_or_create_graph(num_nodes, density, problem_idx):
    directory_name = f'experiments/nodes{num_nodes}/density{density}/problem{problem_idx:02}/'
    definition_file = directory_name + "graph_definition.json"

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    if os.path.exists(definition_file):
        G = create_graph_from_edges(definition_file, num_nodes)
    else:
        G = create_new_graph(definition_file, num_nodes, density)

    return G

def get_IBMQ_topology():
    """
    IBMQ_Montreal gateset
    """
    backend = FakeMontreal()
    topology = Topology(nbqbits=27)

    for g in backend.properties().gates:
        if g.gate == 'cx':
            topology.add_edge(g.qubits[0], g.qubits[1])

    return topology

def get_IBMQ_gateset():
    # gateset = GateSet(['X', 'RZ', 'CNOT', 'SX'])
    gateset = GateSet(['U3', 'CNOT'])
    return gateset

def get_IBMQ_gates_spec():
    return DefaultGatesSpecification()

def get_IBMQ_gate_noise():
    return None

def get_IBMQ_idle_noise():
    return None

def create_QAOA_QLM_job(qubo, params, reps=1, nbshots=512):
    job = qubo.qaoa_ansatz(reps, nbshots=nbshots)
    var_dict = {par_name: par_val for (par_name, par_val) in zip(job.get_variables(), params)}
    job = job(** var_dict)
    return job

def make_pauli_hw_model(direction, single_qubit_error, two_qubit_error=None, circuit=None):
    if circuit is None:
        raise KeyError(f"Circuit must be specified for Pauli-Error")
    if two_qubit_error is None:
        two_qubit_error = single_qubit_error

    if direction == "x":
        err_mat = np.array([[0,1], [1,0]])
    elif direction == "y":
        err_mat = np.array([[0,-1j], [1j,0]])
    elif direction == "z":
        err_mat = np.array([[1,0], [0,-1]])
    else:
        raise KeyError(f"Unknown Pauli gate: {direction}")

    kraus_op = [np.sqrt(single_qubit_error)*err_mat, np.sqrt(1-single_qubit_error)*np.identity(2)]
    single_qubit_noise = QuantumChannelKraus(kraus_op)
    two_qubit_kraus_op = [np.sqrt(two_qubit_error)*err_mat, np.sqrt(1-two_qubit_error)*np.identity(2)]
    two_qubit_noise = QuantumChannelKraus([np.kron(k1, k2) \
            for k1, k2 in product(two_qubit_kraus_op, two_qubit_kraus_op)])

    quantum_channels = dict()

    for gate in circuit.gate_set:
        if circuit.count(gate) > 0:
            if circuit.gate_set[gate].arity == 1:
                if len(circuit.gate_set[gate].parameters) == 0:
                    quantum_channels[gate] = lambda: single_qubit_noise
                else:
                    quantum_channels[gate] = lambda _: single_qubit_noise
            elif circuit.gate_set[gate].arity == 2:
                if len(circuit.gate_set[gate].parameters) == 0:
                    quantum_channels[gate] = lambda: two_qubit_noise
                else:
                    quantum_channels[gate] = lambda _: two_qubit_noise

    return HardwareModel(DefaultGatesSpecification(), quantum_channels, idle_noise = None)


def get_hw_model(single_qubit_error=0.01, two_qubit_error=0.01, noise_type="depolarizing", circuit = None):
    if noise_type == "depolarizing":
        return make_depolarizing_hardware_model(eps1=single_qubit_error, eps2=two_qubit_error)
    elif noise_type == "pauli-x":
        return make_pauli_hw_model("x", single_qubit_error, two_qubit_error, circuit)
    elif noise_type == "pauli-y":
        return make_pauli_hw_model("y", single_qubit_error, two_qubit_error, circuit)
    elif noise_type == "pauli-z":
        return make_pauli_hw_model("z", single_qubit_error, two_qubit_error, circuit)
    else:
        raise KeyError(f"Undefined noise model: {noise_type}")

def get_expectation_QLM(qubo, p, shots=512, simulator="ideal", sim_method="deterministic", single_qubit_noise_prob=0, two_qubit_noise_prob=0):

    best_results = []

    def execute_circ(params):

        # create QAOA circuit
        job = create_QAOA_QLM_job(qubo, params, p, shots)

        if simulator == "ideal":
            qpu = LinAlg()
            stack = qpu
        elif simulator == "mps":
            qpu = MPS(lnnize=True)
            stack = qpu
        else:
            if simulator == "pauli-x-noise":
                hw_model = get_hw_model(single_qubit_noise_prob, two_qubit_noise_prob, "pauli-x", job.circuit)
            elif simulator == "pauli-y-noise":
                hw_model = get_hw_model(single_qubit_noise_prob, two_qubit_noise_prob, "pauli-y", job.circuit)
            elif simulator == "pauli-z-noise":
                hw_model = get_hw_model(single_qubit_noise_prob, two_qubit_noise_prob, "pauli-z", job.circuit)
            elif simulator == "depolarizing-noise":
                hw_model = get_hw_model(single_qubit_noise_prob, two_qubit_noise_prob, "depolarizing")
            else:
                exit(f'Invalid simulator: {simulator}')

            if sim_method == 'stochastic':
                qpu = NoisyQProc(hardware_model=hw_model, sim_method = "stochastic", n_samples = 1000)
                stack = qpu
            elif sim_method == "deterministic":
                qpu = NoisyQProc(hardware_model=hw_model, sim_method = "deterministic-vectorized")
                stack = qpu
            elif sim_method == "experimental":
                qpu = NoisyQProc(hardware_model=hw_model, sim_method = "deterministic-vectorized")
                hw_specs = HardwareSpecs(27, topology = get_IBMQ_topology())
                gateset_compiler = NISQCompiler(target_gate_set=get_IBMQ_gateset())
                transpiler = QuameleonPlugin(specs = hw_specs)
                stack = Nnizer() | gateset_compiler | transpiler | qpu
            else:
                exit(f'Invalid simulation method: {sim_method}')

        result = stack.submit(job)

        print(result.value)
        expectation = result.value

        if len(best_results) == 0:
            best_results.append(params.tolist())
            best_results.append(expectation)
        elif best_results[1] > expectation:
            best_results[0] = params.tolist()
            best_results[1] = expectation

        return expectation

    return execute_circ, best_results

def run_MaxCutQAOA_QLM(density=0.5, num_nodes=20, p=1, simulator="ideal", problem_idx=0, initial_params=None, sim_method="stochastic", single_qubit_noise_prob=0, two_qubit_noise_prob=0):
    G = load_or_create_graph(num_nodes, density, problem_idx)
    max_cut_problem = MaxCut(G)

    # get QUBO
    qubo = max_cut_problem.to_qubo()

    # initialize params
    if initial_params is None:
        initial_params = np.ones(2*p)

    expectation_fkt, result = get_expectation_QLM(qubo, p, simulator=simulator, sim_method=sim_method, single_qubit_noise_prob=single_qubit_noise_prob, two_qubit_noise_prob=two_qubit_noise_prob)

    start = time()
    res = minimize(fun=expectation_fkt,
                    x0=initial_params,
                    method='COBYLA')
    end = time()
    elapsed_time = end - start

    optimal_params = result[0]
    minimum_expectation = result[1]

    print(f"Minimum expectation: {minimum_expectation}")
    print(f"Simulation took: {elapsed_time} s")

    log_result(optimal_params, minimum_expectation, elapsed_time, num_nodes, density, p, problem_idx, simulator, single_qubit_noise_prob, two_qubit_noise_prob, sim_method)

    return optimal_params

def run_MaxCutQAOA_Qiskit(density=0.5, num_nodes=4, p=1, problem_indices=[0]):
    pass

def solve_exact(num_nodes, density, i):
    G = load_or_create_graph(num_nodes, density, i)
    max_cut_problem = MaxCut(G)

    # get QUBO
    qubo = max_cut_problem.to_qubo()
    Q, _ = qubo.get_q_and_offset()

    model = Model('docplex_model')
    v = model.binary_var_list(len(Q))

    A = model.sum(-1*Q[i,i]*v[i] for i in range(len(Q)))
    B = model.sum(-1*Q[i,j]*v[i]*v[j] for i in range(len(Q)) for j in range(len(Q)) if i!=j)

    model.minimize((A + B))

    qubo = from_docplex_mp(model)

    exact_meas = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_meas)
    exact_result = exact.solve(qubo)
    print(exact_result)

def init_params(p, params=None):
    if params is None:
        params = np.ones(2*p)
    else:
        gamma_idx = len(params)//2
        for _ in range(p - gamma_idx):
            params.insert(gamma_idx, 0.0) # beta
            params.append(0.0) # gamma
    return params

def run_MaxCutQAOA_QLM_for_multiple_p(num_nodes, density, p_range, simulator, problem_idx, sim_method, single_noise_prob, two_noise_prob):
    old_params = None
    for p in p_range:
        print(f"p={p}")
        initial_params = init_params(p, old_params)
        old_params = run_MaxCutQAOA_QLM(num_nodes=num_nodes, density=density, p=p, simulator=simulator, problem_idx=problem_idx, initial_params=initial_params, sim_method=sim_method, single_qubit_noise_prob=single_noise_prob, two_qubit_noise_prob= two_noise_prob)

if __name__ == '__main__':
    pool = Pool()
    for i in range(1,4):
        for prob in np.linspace(0.00, 0.05, 6):
            for noise in ["pauli-x-noise", "pauli-y-noise", "pauli-z-noise", "depolarizing-noise"]:
                pool.apply_async(run_MaxCutQAOA_QLM_for_multiple_p, args=(5, 0.5, range(1,16), noise, i, "deterministic", prob, prob))
        # pool.apply_async(run_MaxCutQAOA_QLM_for_multiple_p, args=(6, 0.4, range(1,16), "ideal", i, "stochastic", 0))
    pool.close()
    pool.join()
    # run_MaxCutQAOA_QLM_for_multiple_p(13, 0.3, range(1,16), "pauli-x-noise", 0, "deterministic", 0.01)
    # solve_exact(6, 0.4, 0)

