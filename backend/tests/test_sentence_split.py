from app.services.sentence_split import RegexSentenceSplitter


def test_splits_simple_sentences():
    splitter = RegexSentenceSplitter()
    result = splitter.split("This is one sentence. This is another sentence.")
    assert result == [
        "This is one sentence.",
        "This is another sentence.",
    ]


def test_protects_common_abbreviations():
    splitter = RegexSentenceSplitter()
    result = splitter.split(
        "Dr. Rao studied U.S. history. He later moved to the U.K. for a postdoc."
    )
    assert len(result) == 2
    assert result[0].startswith("Dr. Rao studied U.S. history.")
    assert result[1].startswith("He later moved to the U.K.")


def test_handles_empty_and_whitespace_only_text():
    splitter = RegexSentenceSplitter()
    assert splitter.split("") == []
    assert splitter.split("   \n\t  ") == []


def test_collapses_internal_whitespace():
    splitter = RegexSentenceSplitter()
    result = splitter.split("This   has\nweird   spacing. Second sentence.")
    assert result[0] == "This has weird spacing."


def test_single_sentence_no_terminal_punctuation():
    splitter = RegexSentenceSplitter()
    result = splitter.split("An incomplete fragment without a period")
    assert result == ["An incomplete fragment without a period"]
