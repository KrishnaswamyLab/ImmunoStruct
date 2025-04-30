from models.ablation_models import *
from models.hybrid_models import *
from models.comparative_models import *


model_map = {
    "SequenceModel": SequenceModel,
    "SequenceFpModel": SequenceFpModel,
    "StructureModel": StructureModel,
    "StructureModel_SSL": StructureModel_SSL,
    "StructureModelv2": StructureModelv2,
    "HybridModel": HybridModel,
    "HybridModel_SSL": HybridModel_SSL,
    "HybridModelv2": HybridModelv2,
    "HybridModelv2_SSL": HybridModelv2_SSL,
    "HybridModel_Comparative": HybridModel_Comparative,
    "HybridModel_Comparative_SSL": HybridModel_Comparative_SSL,
    "HybridModelv2_Comparative": HybridModelv2_Comparative,
    "HybridModelv2_Comparative_SSL": HybridModelv2_Comparative_SSL,
    "DualModel": DualModel,
}