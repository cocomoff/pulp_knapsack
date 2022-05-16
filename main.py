from solver import *

if __name__ == '__main__':
    weights = [4, 2, 2, 1, 10]
    profits = [12, 2, 1, 1, 4]
    capacity = 15

    # greedy
    solution_g = solveKnapsackGreedily(profits, weights, capacity)
    print(solution_g)

    # or-tools
    solution_o = solveKnapsackProblem(profits, weights, capacity)
    print(solution_o)

    # relax
    solution_r = solveKnapsackProblemRelaxation(profits, weights, capacity)
    print(solution_r)

    # exact
    solution_e = solveKnapsackProblemExact(profits, weights, capacity)
    print(solution_e)
