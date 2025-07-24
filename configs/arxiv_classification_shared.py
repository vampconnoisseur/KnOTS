TASK_A_SUBSET = [
    'cs.CV', 'cs.AI', 'cs.CL', 'cs.RO', 'cs.CR', 'cs.NE'
]

TASK_B_SUBSET = [
    'cs.LG', 'stat.ML', 'math.OC', 'eess.SP', 'math.ST', 'stat.TH'
]

ALL_LABELS = sorted(list(set(
    TASK_A_SUBSET +
    TASK_B_SUBSET
)))

LABEL_TO_ID = {name: i for i, name in enumerate(ALL_LABELS)}
ID_TO_LABEL = {i: name for i, name in enumerate(ALL_LABELS)}
NUM_LABELS = len(ALL_LABELS)