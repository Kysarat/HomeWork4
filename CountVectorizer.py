class CountVectorizer:
    """
    Конвертирует слова в вектора

    """
    def __init__(self):
        self.unique_words = set()

    def get_feature_names(self) -> list:
        return list(self.unique_words)

    def fit_transform(self, corpus: list) -> list:
        if not isinstance(corpus, list):
            raise TypeError('Incorrect type of the input data')

        for string in corpus:
            list_split = string.split()
            for word in list_split:
                self.unique_words.add(word.lower())

        token2id = {token: i for i, token in enumerate(self.unique_words)}
        list_embedding = []
        for i in range(len(corpus)):
            list_embedding.append([0 for num in range(len(token2id))])

        for i in range(len(corpus)):
            sentence_list = corpus[i].split()
            for word in sentence_list:
                list_embedding[i][token2id[word.lower()]] += 1
        return list_embedding
