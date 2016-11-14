import numpy as np
import nltk
import pandas as pd
from ast import literal_eval
from collections import Counter


def sampleFromDirichlet(alpha):
    return np.random.dirichlet(alpha)


def sampleFromCategorical(theta):
    # theta = theta / np.sum(theta)
    return np.random.multinomial(1, theta).argmax()


def word_indices(doc_sent_word_dict, sent_index):
    """
    :param doc_sent_word_dict:
    :param sent_index:
    :return:
    """
    sentence = doc_sent_word_dict[sent_index]
    for idx in sentence:
        yield idx


class STMD_Gibbs_Sampler:
    def __init__(self, numTopics, alpha, beta, gamma, max_vocab_size=10000, max_sentence=50, numSentiments=2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numTopics = numTopics
        self.numSentiments = numSentiments
        self.MAX_VOCAB_SIZE = max_vocab_size
        self.maxSentence = max_sentence

    def build_dataset(self, reviews):
        """
        :param reviews: 리뷰 데이터 [ [[문서1의 문장1],[문서1의 문장2]], [[문서2의 문장1],[문서2의 문장2]], ...]]
        :return:
        """
        corpus = [word for review in reviews for sentence in review for word in sentence]
        text = nltk.Text(corpus)
        freq = nltk.FreqDist(text)
        keywords = [tup[0] for tup in freq.most_common(self.MAX_VOCAB_SIZE)]  # 많이 등장한 단어 선택

        word2idx = {}  # key : 단어, value : index
        for index, key in enumerate(keywords):
            word2idx[key] = index

        idx2word = dict(zip(word2idx.values(), word2idx.keys()))  # key : index, value : 단어
        doc_sent_word_dict = {}  # key: 문서 index, value : [[list of sent1 단어의 index], [list of sent2 단어의 index]...]
        numSentence = {}  # key : 문서 index, value : 해당 문서의 문장수
        wordCountSentence = {}  # key : 문서 index, value : 해당 문서의 각 문장별 word count
        for index, review in enumerate(reviews):
            doc_sent_lst = []
            doc_sent_count = []
            for sent in review:
                word_indices = [word2idx[word] for word in sent if word in word2idx]
                doc_sent_lst.append(word_indices)
                counts = Counter(word_indices)
                doc_sent_count.append(counts)
            numSentence[index] = len(doc_sent_lst)
            doc_sent_word_dict[index] = doc_sent_lst
            wordCountSentence[index] = doc_sent_count

        return word2idx, idx2word, doc_sent_word_dict, wordCountSentence, numSentence

    def _initialize_(self, reviews):
        self.word2idx, self.idx2word, self.doc_sent_word_dict, self.wordCountSentence, self.numSentence = self.build_dataset(
            reviews)
        numDocs = len(self.doc_sent_word_dict.keys())
        vocabSize = len(self.word2idx.keys())

        # Pseudocounts
        self.n_wkl = np.zeros((vocabSize, self.numTopics, self.numSentiments))  # 단어 i가 topic k, senti l로 할당된 수
        self.n_kl = np.zeros((self.numTopics, self.numSentiments))  # topic k, senti l로 할당된 단어 수
        self.ns_d = np.zeros((numDocs))  # 문서 d의 문장 수
        self.ns_dkl = np.zeros((numDocs, self.numTopics, self.numSentiments))  # 문서 d에서 topic k, sentiment l로 할당된 문장 수
        self.ns_dk = np.zeros((numDocs, self.numTopics))  # 문서 d에서 topic k로 할당된 문장 수
        self.topics = {}
        self.sentiments = {}
        # self.priorSentiment = {}

        alphaVec = self.alpha * np.ones(self.numTopics)
        gammaVec = self.gamma * np.ones(self.numSentiments)
        # 기존 sentiment-lda에서는 sentiment wordnet을 이용해서 priorsentiment를 줬는데,
        # word2vec은 classvector와 유사성을 이용해서 해도 괜찮을듯
        #         for i, word in enumerate(self.vectorizer.get_feature_names()):
        #             synsets = swn.senti_synsets(word)
        #             posScore = np.mean([s.pos_score() for s in synsets])
        #             negScore = np.mean([s.neg_score() for s in synsets])
        #             if posScore >= 0.1 and posScore > negScore:
        #                 self.priorSentiment[i] = 1
        #             elif negScore >= 0.1 and negScore > posScore:
        #                 self.priorSentiment[i] = 0

        for d in range(numDocs):
            topicDistribution = sampleFromDirichlet(alphaVec)
            sentimentDistribution = np.zeros((self.numTopics, self.numSentiments))

            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)

            for m in range(self.numSentence[d]):
                t = sampleFromCategorical(topicDistribution)
                s = sampleFromCategorical(sentimentDistribution[t, :])
                self.topics[(d, m)] = t  # d 문서의 m번째 문장의 topic
                self.sentiments[(d, m)] = s  # d 문서의 m 번째 문장의 sentiment
                self.ns_d[d] += 1
                self.ns_dkl[d, t, s] += 1
                self.ns_dk[d, t] += 1
                for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):  # d번째 문서의 m번째 문장의 단어를 돌면서
                    self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                    self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수

    def conditionalDistribution(self, d, m, w):
        """
        Calculates the (topic, sentiment) probability for sentence m in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))

        # firstfactor 수정
        firstFactor = (self.n_wkl[w, :, :] + self.beta) / \
                      (self.n_kl + self.n_wkl.shape[0] * self.beta)  # dim(K x L)

        secondFactor = (self.ns_dk[d, :] + self.alpha) / \
                       (self.ns_d[d] + self.numTopics * self.alpha)  # dim(K x 1)

        thirdFactor = (self.ns_dkl[d, :, :] + self.gamma) / \
                      (self.ns_dk[d] + self.numSentiments * self.gamma)[:, np.newaxis]  # dim (K x L)

        probabilities_ts *= firstFactor * thirdFactor
        probabilities_ts *= secondFactor[:, np.newaxis]
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

    def run(self, reviews, maxIters=10):
        self._initialize_(reviews)
        numDocs = len(self.doc_sent_word_dict.keys())

        for iteration in range(maxIters):
            if (iteration + 1) % 10 == 0:
                print("Starting iteration %d of %d" % (iteration + 1, maxIters))
            for d in range(numDocs):
                for m in range(self.numSentence[d]):
                    t = self.topics[(d, m)]
                    s = self.sentiments[(d, m)]
                    self.ns_d[d] -= 1
                    self.ns_dkl[d, t, s] -= 1
                    self.ns_dk[d, t] -= 1
                    for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
                        self.n_wkl[w, t, s] -= 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                        self.n_kl[t, s] -= 1  # topic k, senti l로 할당된 단어 수

                    probabilities_ts = self.conditionalDistribution(d, m, w)
                    ind = sampleFromCategorical(probabilities_ts.flatten())
                    t, s = np.unravel_index(ind, probabilities_ts.shape)
                    self.topics[(d, m)] = t
                    self.sentiments[(d, m)] = s
                    self.ns_d[d] += 1
                    self.ns_dkl[d, t, s] += 1
                    self.ns_dk[d, t] += 1
                    for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
                        self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                        self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수


data = pd.read_csv("E:/dataset/MasterThesis/elec_df_brand2vec.csv",nrows =1000)
data['reviewSentence'] = data.reviewSentence.apply(lambda row: literal_eval(row))
data['reviewSentence_tagged'] = data.reviewSentence_tagged.apply(lambda row: literal_eval(row))
tagged_text_list = list(data['reviewSentence_tagged'])
sampler = STMD_Gibbs_Sampler(numTopics=5, alpha=0.1, beta=0.1, gamma=0.5, numSentiments=2)
sampler._initialize_(tagged_text_list)
sampler.run(tagged_text_list)
print("end")