from inkbox_claude.signoff import strip_signoff

T = "Executive Assistant to Aaron Blatnoy"


def test_strips_name_and_title():
    assert strip_signoff(f"Thanks! I will confirm that with Aaron.\n\nBlatbot\n{T}") == "Thanks! I will confirm that with Aaron."


def test_strips_with_closing_word():
    assert strip_signoff(f"See you at 3.\n\nBest,\nBlatbot\n{T}") == "See you at 3."


def test_strips_one_line_form():
    assert strip_signoff(f"Done.\nBlatbot / {T}") == "Done."


def test_leaves_ordinary_text_alone():
    assert strip_signoff("Best,\nsee you then") == "Best,\nsee you then"
    assert strip_signoff(f"I am the {T} and can help.") == f"I am the {T} and can help."


def test_never_returns_empty():
    assert strip_signoff(f"Blatbot\n{T}") == f"Blatbot\n{T}"
