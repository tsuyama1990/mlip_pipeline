from enum import Enum


class StructureType(str, Enum):
    SQS = "sqs"
    STRAIN = "strain"
    RATTLE = "rattle"
    VACANCY = "vacancy"
    INTERSTITIAL = "interstitial"
    BASE = "base"
    MOLECULE = "molecule"
