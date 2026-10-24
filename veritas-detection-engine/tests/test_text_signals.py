from src.text.signals import burstiness


def test_uniform_sentences_have_zero_burstiness():
    assert burstiness("One two three. Four five six. Seven eight nine.") == 0.0


def test_varied_sentences_have_higher_burstiness():
    varied = "Short. This one is quite a bit longer than the first. Ok."
    assert burstiness(varied) > 0.5


def test_single_sentence_returns_zero():
    assert burstiness("Just one sentence.") == 0.0
