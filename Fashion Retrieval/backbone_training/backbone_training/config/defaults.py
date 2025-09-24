from yacs.config import CfgNode as CN
import logging

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# Dataset option
_C.DATA = CN()
_C.DATA.NUM_WORKERS = 4
_C.DATA.NUM_TRIPLETS = 100000
_C.DATA.TRAIN_BATCHSIZE = 32
_C.DATA.TEST_BATCHSIZE = 64
_C.DATA.BASE_PATH = "/xlearning/honglin/datasets"
_C.DATA.DATASET = ""
_C.DATA.NUM_ATTRIBUTES = -1

# Attributes, refer to specific dataset configuration file
_C.DATA.ATTRIBUTES = CN(new_allowed=True)

_C.DATA.PATH_FILE = CN()
_C.DATA.PATH_FILE.TRAIN = ""
_C.DATA.PATH_FILE.VALID = ""
_C.DATA.PATH_FILE.TEST = ""

_C.DATA.GROUNDTRUTH = CN(new_allowed=True)

_C.DATA.GROUNDTRUTH.QUERY = CN()
_C.DATA.GROUNDTRUTH.QUERY.TEST = ""
_C.DATA.GROUNDTRUTH.QUERY.VALID = ""

_C.DATA.GROUNDTRUTH.CANDIDATE = CN()
_C.DATA.GROUNDTRUTH.CANDIDATE.TEST = ""
_C.DATA.GROUNDTRUTH.CANDIDATE.VALID = ""

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.EVAL_STEPS = 1
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STEP_SIZE = 3
_C.SOLVER.DECAY_RATE = 0.9
_C.SOLVER.EPOCHS = 50
_C.SOLVER.LOG_PERIOD = 800
_C.SOLVER.CLIP_LR = 1e-6
# _C.SOLVER.MLP_LR = 1e-5
_C.SOLVER.MARGIN = 0.3
# _C.SOLVER.BETA = 0.6

_C.DEVICE= "cuda"

# Logger
_C.LOGGER = CN()
_C.LOGGER.LEVEL = logging.INFO
_C.LOGGER.STREAM = "stdout"