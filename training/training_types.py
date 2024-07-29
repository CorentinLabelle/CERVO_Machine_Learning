from enum import Enum


class TrainingType(Enum):

    CROSS_VALIDATION = "cross-validation"
    TIME_GENERALIZATION = "time generalization"

    def tag(self):
        if self == TrainingType.CROSS_VALIDATION:
            return "cross_validation"
        elif self == TrainingType.TIME_GENERALIZATION:
            return "time_generalization"

    def title(self):
        if self == TrainingType.CROSS_VALIDATION:
            return "Cross-Validation"
        elif self == TrainingType.TIME_GENERALIZATION:
            return "Time Generalization"
