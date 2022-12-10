from dataclasses import dataclass


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: float
    end: float
    score: float

    def __repr__(self):
        return (
            f"{self.label}\t({self.score:4.2f}): [{self.start:0.2f}, {self.end:0.2f})"
        )

    @property
    def length(self):
        return self.end - self.start


def segment_to_word(segment: Segment, ratio=0.02):
    # assert isinstance(segment, Segment)
    return [segment.label, segment.start * ratio, segment.end * ratio]
