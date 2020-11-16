from CountVectorizer import CountVectorizer
from TfidfTransformer import TfidfTransformer


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.transformer = TfidfTransformer()

    def fit_transform(self, corpus: list):
        count_matrix = super().fit_transform(corpus)
        return self.transformer.fit_transform(count_matrix)
