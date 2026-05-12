from aequilibrae.utils.aeq_signal import simple_progress


class EmitCapture:
    """A signal like object that just captures any emits that are given to it"""

    def __init__(self):
        self.emits = []

    def emit(self, e):
        self.emits.append(e)


def test_simple_progress():
    emit_capturer = EmitCapture()
    for i in simple_progress([1, 2, 3], emit_capturer):
        print(i)
    assert emit_capturer.emits == [
        ["start", 3, "0/3"],
        ["update", 1, "1/3"],
        ["update", 2, "2/3"],
        ["update", 3, "3/3"],
    ]
