from util import read_mat
from typing import List


class BstStudy:

    def __init__(self, brainstorm_study_path: str):
        self.file_path: str = brainstorm_study_path

        raw_brainstorm_study_file = read_mat.with_scipy(self.file_path, with_key=False)

        self.name: str = raw_brainstorm_study_file["Name"][0]

        self.date: str = raw_brainstorm_study_file["DateOfStudy"][0]

        self.bad_trials = []
        for iBadTrial in range(raw_brainstorm_study_file["BadTrials"].size):
            self.bad_trials.append(str(raw_brainstorm_study_file["BadTrials"][iBadTrial][0][0]))

    def get_bad_trials(self) -> List[str]:
        return self.bad_trials

    def __str__(self) -> str:
        return f"File: {self.file_path}\n" \
               f"Name: {self.name}\n" \
               f"Date: {self.date}\n" \
               f"Number of bad trials: {len(self.bad_trials)}"


if __name__ == "__main__":
    path = \
        "/mnt/3b5a15cf-20ff-4840-8d84-ddbd428344e9/ALAB1/rg/bs_db/" \
        "Loca_intra_AUDI/data/HEJ_Subject01/1/brainstormstudy.mat"

    brainstorm_study = BstStudy(path)

    print(f"Raw brainstorm study:\n{brainstorm_study}\n")
