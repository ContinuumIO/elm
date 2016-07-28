import array

from deap import creator, base, tools
from deap.tools.emo import selNSGA2

def pareto_front(weights, objectives, take=None):
    toolbox = base.Toolbox()
    take = take or objectives.shape[0]
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual",
                   array.array,
                   typecode='d',
                   fitness=creator.FitnessMulti)
    toolbox.register('evaluate', lambda x: x)
    objectives = [creator.Individual(objectives[idx, :])
                  for idx in range(objectives.shape[0])]
    for (idx, obj) in enumerate(objectives):
        obj.idx = idx
        obj.fitness.values = toolbox.evaluate(obj)
    sel = selNSGA2(objectives, take)
    return tuple(item.idx for item in sel)
