from unittest import TestCase

from UNF.data.tokenizer import WhitespaceTokenizer, SpacyTokenizer

class TestToken(TestCase):

    def test_whitespacetokenizer(self):
        self.assertEqual(WhitespaceTokenizer()("a b c"), ["a", "b", "c"])
