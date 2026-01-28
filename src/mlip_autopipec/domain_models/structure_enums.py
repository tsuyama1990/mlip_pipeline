from enum import Enum


class StructureType(str, Enum):
    SQS = "sqs"
    STRAIN = "strain"
    RATTLE = "rattle"
    VACANCY = "vacancy"
    INTERSTITIAL = "interstitial"
    BASE = "base"
    MOLECULE = "molecule"


class CandidateStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    FAILED = "failed"
    SCREENING = "screening"
    REJECTED = "rejected"
    COMPLETED = "completed"
