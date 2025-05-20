from conn2res.tasks import NeuroGymTask, ReservoirPyTask, Conn2ResTask

NEUROGYM_TASKS = (
    'AntiReach',
    'ContextDecisionMaking',
    'DelayComparison',
    'DelayMatchCategory',
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    'DualDelayMatchSample',
    'GoNogo',
    'HierarchicalReasoning',
    'IntervalDiscrimination',
    'MotorTiming',
    'MultiSensoryIntegration',
    'OneTwoThreeGo',
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'PostDecisionWager',
    'ProbabilisticReasoning',
    'PulseDecisionMaking',
    'Reaching1D',
    'Reaching1DWithSelfDistraction',
    'ReadySetGo',
    'SingleContextDecisionMaking',
)
RESERVOIRPY_TASKS = (
    'henon_map',
    'lorenz',
    'multiscroll',
    'doublescroll',
    'rabinovich_fabrikant',
    'lorenz96',
    'rossler'
)
CONN2RES_TASKS = (
    'MemoryCapacity',
)
def res_task(task_name, task_type: str):
    if task_type == 'neurogym':
        return NeuroGymTask(name=task_name)
    elif task_type == 'reservoirpy':
        return ReservoirPyTask(name=task_name)
    elif task_type == 'conn2res':
        return Conn2ResTask(name=task_name)
    else:
        raise ValueError('task_type must be one of the following: neurogym, reservoirpy, conn2res')