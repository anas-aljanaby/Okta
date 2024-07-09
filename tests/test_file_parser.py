import os
import io
import pytest
import file_parser
from file_parser import parse_file  # Replace with your actual import


def create_file_like_object(file_path, file_type):
    with open(file_path, "rb") as f:
        file_content = f.read()
    file_like_object = io.BytesIO(file_content)
    file_like_object.name = file_path
    file_like_object.type = file_type
    return file_like_object


# Define a fixture for the base directory of test files
@pytest.fixture
def base_dir():
    return os.path.join(os.path.dirname(__file__), "test_files")


english_test_phrase = "This is a test document in English."
arabic_test_phrase = ".ﺔﻴﺑﺮﻌﻟﺍ ﺔﻐﻠﻟﺎﺑ ﺭﺎﺒﺘﺧﺍ ﺔﻘﻴﺛﻭ ﻩﺬﻫ"
# Define the test cases
test_cases = [
    ("test_english.pdf", english_test_phrase, "application/pdf"),
    ("test_arabic.pdf", arabic_test_phrase, "application/pdf"),
    (
        "test_english.docx",
        english_test_phrase,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    # ("test_arabic.docx", arabic_test_phrase, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (
        "test_english.pptx",
        english_test_phrase,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    (
        "test_arabic.pptx",
        arabic_test_phrase,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ("test_english.txt", english_test_phrase, "text/plain"),
    # ("test_arabic.txt", arabic_test_phrase, "text/plain"),
]


def clean_whitespace(text):
    return " ".join(text.split())


@pytest.mark.parametrize("file_name,expected_text,file_type", test_cases)
def test_parse_file(base_dir, file_name, expected_text, file_type):
    file_path = os.path.join(base_dir, file_name)
    file_like_object = create_file_like_object(file_path, file_type)
    extracted_text = clean_whitespace(parse_file(file_like_object))

    assert expected_text in extracted_text, f"Test failed for {file_name}"
