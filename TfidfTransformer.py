import math


class TfidfTransformer:
    """
    Transform a count matrix to tf-idf representation

    """

    def tf_transform(self, matrix: list):
        """
        Count term frequency
        """
        tf_matrix = []
        for lis in matrix:
            number = sum(lis)
            vector = [round(value / number, 3) for value in lis]
            tf_matrix.append(vector)
        return tf_matrix

    def idf_transform(self, matrix: list):
        """
        The inverse document frequency (IDF) vector
        """
        number = len(matrix)
        quan_matr = len(matrix[0])
        idf_matrix = [0] * quan_matr
        for i in range(quan_matr):
            for line in matrix:
                if line[i] > 0:
                    idf_matrix[i] += 1
        idf = [
            round((math.log((number + 1) / (document + 1)) + 1), 3)
            for document in idf_matrix
        ]
        return idf

    def _tfidf_transform(self, tf_matrix: list, idf: list):
        """
        Compute tf-idf transformation
        """
        tf_idf = []
        for line in tf_matrix:
            tf_idf.append([round(line[i] * idf[i], 3)
                           for i in range(len(idf))])
        return tf_idf

    def fit_transform(self, matrix: list):
        """
        Display tf-idf representation
        """
        return self._tfidf_transform(self.tf_transform(matrix),
                                     self.idf_transform(matrix))
