from ortools.algorithms import pywrapknapsack_solver
import time
from math import ceil
import pulp


def solveKnapsackGreedily(profits, weights, capacity):
    assert len(profits) == len(weights)
    assert len([x for x in profits if x < 0]) == 0
    assert (capacity > 0 and len(profits) >= 2)

    n_items = len(profits)
    items = [[profits[i], weights[i], i] for i in range(n_items)]
    items.sort(key=lambda x: float(x[0])/x[1], reverse=True)
    assert(items[0][0]/items[0][1] >= items[1][0]/items[1][1])

    objective = 0
    start_time = time.time()
    available_capacity = capacity
    assignments = [0 for i in range(n_items)]
    for i in range(n_items):
        if items[i][1] > available_capacity:
            continue
        objective += items[i][0]  # profit increase
        available_capacity -= items[i][1]  # weight
        assignments[items[i][2]] = 1

        if available_capacity == 0:
            break
    solution_info = {'objective': objective,
                     'assignments': assignments, 'runtime': time.time() - start_time}
    return solution_info


def solveKnapsackProblem(profits, weights, capacity):
    """
    see https://developers.google.com/optimization/bin/knapsack
    """
    solver_type = pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER
    solver = pywrapknapsack_solver.KnapsackSolver(solver_type, "ortools_kn")
    weights = [weights]
    capacity = [capacity]
    start_time = time.time()
    solution_info = {}

    try:
        solver.Init(profits, weights, capacity)
        val = solver.Solve()
        solution_info['objective'] = val
        solution_info['assignments'] = [
            int(solver.BestSolutionContains(x)) for x in range(len(profits))]
        solution_info['runtime'] = time.time() - start_time

    except:
        solution_info['objective'] = 0
        solution_info['assignments'] = [0 for x in range(len(profits))]
        solution_info['runtime'] = 0
    return solution_info


def solveKnapsackProblemRelaxation(profits, weights, capacity):
    # profits = [v for v in profits]
    # weights = [[w for w in W] for W in weights]
    # capacity = [c for c in capacity]
    start_time = time.time()
    n = len(profits)
    m = pulp.LpProblem("Relaxed-Knapsack", sense=pulp.LpMaximize)
    x = {}
    for i in range(n):
        x[i] = pulp.LpVariable(f"x-{i}", 0, 1, cat=pulp.LpInteger)
    m += pulp.lpSum(x[i] * profits[i] for i in range(n))
    m += pulp.lpSum(x[i] * weights[i] for i in range(n)) <= capacity
    status = m.solve(pulp.PULP_CBC_CMD(msg=0))

    solution_info = {}
    if pulp.LpStatus[status] == "Optimal":
        solution_info['objective'] = m.objective.value()
        sol = [0] * n
        for i in range(n):
            if x[i].value() > 0:
                sol[i] = 1
        solution_info['assignments'] = sol
        solution_info['runtime'] = time.time() - start_time
    else:
        solution_info['objective'] = 0
        solution_info['assignments'] = [0 for x in range(len(profits))]
        solution_info['runtime'] = 0
    return solution_info


def solveKnapsackProblemExact(profits, weights, capacity):
    # profits = [v for v in profits]
    # weights = [[w for w in W] for W in weights]
    # capacity = [c for c in capacity]
    start_time = time.time()
    n = len(profits)
    m = pulp.LpProblem("Relaxed-Knapsack", sense=pulp.LpMaximize)
    x = {}
    for i in range(n):
        x[i] = pulp.LpVariable(f"x-{i}", cat=pulp.LpBinary)
    m += pulp.lpSum(x[i] * profits[i] for i in range(n))
    m += pulp.lpSum(x[i] * weights[i] for i in range(n)) <= capacity
    status = m.solve(pulp.PULP_CBC_CMD(msg=0))

    solution_info = {}
    if pulp.LpStatus[status] == "Optimal":
        solution_info['objective'] = ceil(m.objective.value())
        sol = [0] * n
        for i in range(n):
            if x[i].value() > 0:
                sol[i] = 1
        solution_info['assignments'] = sol
        solution_info['runtime'] = time.time() - start_time
    else:
        solution_info['objective'] = 0
        solution_info['assignments'] = [0 for x in range(len(profits))]
        solution_info['runtime'] = 0

    return solution_info
