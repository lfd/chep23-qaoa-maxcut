import os
import json
import networkx as nx
import numpy as np
from time import time
#import logging
#logging.basicConfig(level=logging.INFO)

from qat.opt.max_cut import MaxCut
from qat.qpus.hook_linalg import LinAlg
from qat.plugins import ScipyMinimizePlugin, QuameleonPlugin, Nnizer
from qat.core.console import display
from qat.opt.circuit_generator import CircuitGenerator
from qat.core import HardwareSpecs, Topology
from qat.core.gate_set import GateSignature, GateSet
from qat.noisy import NoisyQProc
from qat.hardware import make_depolarizing_hardware_model

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

def log_result(result, elapsed_time, num_nodes, density, p, i, simulator):
    directory_name = f'experiments/nodes{num_nodes}/density{density}/problem{i:02}/Results/{simulator}/p{p:02}/'

    if os.path.exists(directory_name):
        print(f'Overwriting results in {directory_name}')
        [os.remove(directory_name + f) for f in os.listdir(directory_name)]
    else:
        print(f'Creating new result directory in {directory_name}')
        os.makedirs(directory_name)


    with open(directory_name + 'optimizer_data.log', 'w') as f:
        f.write(result.meta_data['optimizer_data'])

    with open(directory_name + 'parameters.json', 'w') as f:
        f.write(json.dumps(result.meta_data['parameter_map']))

    with open(directory_name + 'optimization_trace.json', 'w') as f:
        f.write(json.dumps(result.meta_data['optimization_trace']))

    with open(directory_name + 'elapsed_time.json', 'w') as f:
        f.write(json.dumps(elapsed_time))

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
    IBMQ_Brooklyn gateset
    """
    backend = FakeMontreal()
    topology = Topology(nbqbits=27)

    for g in backend.properties().gates:
        if g.gate == 'cx':
            topology.add_edge(g.qubits[0], g.qubits[1])

    return topology

def get_IBMQ_gateset():
    gateset = GateSet(['X', 'RZ', 'CNOT', 'SX'])
    return gateset

def run_MaxCutQAOA_QLM(density=0.5, num_nodes=20, p=1, simulator="ideal", problem_indices=[0]):
    for i in problem_indices:
        G = load_or_create_graph(num_nodes, density, i)
        max_cut_problem = MaxCut(G)

        # get QUBO
        qubo = max_cut_problem.to_qubo()

        # create QAOA circuit
        job = qubo.qaoa_ansatz(p, nbshots=512)

        # hw_specs = HardwareSpecs(27, topology = get_IBMQ_topology(), gateset = get_IBMQ_gateset())
        hw_specs = HardwareSpecs(27, topology = get_IBMQ_topology())
        transpiler = QuameleonPlugin(specs = hw_specs)
        hw_model = make_depolarizing_hardware_model(eps1=0.005, eps2=0.012)

        scipy_plugin = ScipyMinimizePlugin(method = "COBYLA")

        if simulator == "ideal":
            qpu = LinAlg()
        elif simulator == "noisy_deterministic":
            qpu = NoisyQProc(hardware_model=hw_model, sim_method = "deterministic-vectorized")
        elif simulator == "noisy_stochastic":
            qpu = NoisyQProc(hardware_model=hw_model, sim_method = "stochastic", n_samples = 10000)
        else:
            exit(f'Invalid simulator: {simulator}')

        stack = scipy_plugin | Nnizer() | transpiler | qpu


        start = time()
        result = stack.submit(job)
        end = time()

        print(result.value)
        print(f"Simulation took: {end-start} s")
        elapsed_time = end - start

        log_result(result, elapsed_time, num_nodes, density, p, i, simulator)

def run_MaxCutQAOA_Qiskit(density=0.5, num_nodes=4, p=1, problem_indices=[0]):
    pass

if __name__ == '__main__':
    for p in range(1, 21):
        print(f"p={p}")
        run_MaxCutQAOA_QLM(num_nodes=10, p=p)

