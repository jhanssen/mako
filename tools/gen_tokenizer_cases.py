"""Writes tokenizer test cases separated by 0x1E (tests/data/glm_tokenizer_cases.bin).

    python3 tools/gen_tokenizer_cases.py > tests/data/glm_tokenizer_cases.bin

The expected ids (tests/data/glm_tokenizer_expected.txt) come from the reference tokenizer through ref_dump
(REF_TOKENIZE=<cases file>).
"""
import random
import sys

cases = [
    "The capital of France is", "", " ", "  ", "\n", "\n\n", " \n ", "a\r\nb", "\t\tindented\n\tmore",
    "Hello, world! How's it going? I'm fine, you're nice, we've been, they'll go, he'd say. IT'S LOUD'S",
    "'s 't 're 've 'm 'll 'd 'x 'S 'RE",
    "1 12 123 1234 12345 1234567890 3.14159 -42 1e10 0x1F 1,000,000",
    "def foo(x):\n    return x * 2  # comment\n\n\nclass Bar:\n\tpass\n",
    "int main() {\n  printf(\"%d\\n\", 42);\n  return 0;\n}\n",
    "multiple   spaces    here     and\ttabs\t\tand   \n   newlines   ",
    "trailing spaces   ", "   leading spaces", "end with newline\n", "a  b", "a   b", "a \n b", "a\n\n\nb",
    "你好，世界！这是一个测试。", "日本語のテキストです。カタカナとひらがな。", "한국어 텍스트입니다.",
    "Ünïcödé àccénts ñ ç ø å æ ß", "Привет, мир! Это тест.", "مرحبا بالعالم", "नमस्ते दुनिया", "สวัสดีชาวโลก",
    "emoji 😀🎉👍🏽 👨‍👩‍👧‍👦 🇫🇷 ❤️", "math ∑∫∂√∞ ≤ ≥ ≠ ± × ÷ π", "nbsp\u00a0here\u3000ideographic\u2009thin",
    "[gMASK]<sop><|system|>\nYou are helpful.<|user|>\nHi<|assistant|>\n<think></think>\nHello!",
    "<|endoftext|>", "<|user|><|user|>", "text<|assistant|>text", "<think>reasoning</think>answer",
    "https://example.com/path?q=1&r=two#frag", "user@example.com", "C:\\Windows\\System32",
    "\"quoted\" 'single' `back` «guillemets» „low“", "...!!!???---___===+++", "a.b.c,d;e:f",
    "ALLCAPS lowercase CamelCase snake_case kebab-case", "x" * 200, " " * 50 + "x", "\n" * 10 + "x",
    "mixed\r\n\r\nline\rendings\n", "١٢٣ ٤٥٦ Arabic-Indic digits ①②③ Ⅻ ½ ¾", "a'b c'd it''s ''",
]
palette = list("abcXYZ019 \t\n\r'\".,!?-_()[]{}<>/\\@#$%^&*") + [
    "é", "ß", "中", "文", "😀", "\u00a0", "\u3000", "١", "½", "<|user|>", "'s", "'ll", "  ", "\n\n"]
random.seed(1234)
for _ in range(2000):
    cases.append("".join(random.choice(palette) for _ in range(random.randint(1, 40))))
sys.stdout.write("\x1e".join(cases))
