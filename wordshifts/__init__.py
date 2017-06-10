import os
import numpy as np
from scipy.stats import norm


class WordShifts(object):

    def __init__(self, methods=['happiness', 'anew'], config={}):
        self.methods = methods
        self.config = config
        self.sentiment = Sentiment(self.config)

    def computeWordShifts(self, docComparison, docReference):
        sentiComparison = self.sentiment.compute(docComparison, self.methods, True)
        sentiReference = self.sentiment.compute(docReference, self.methods, True)

        wordShifts = dict()
        if 'happiness' in self.methods:
            wordShifts['happiness'] = dict()
        if 'anew' in self.methods:
            for m in ['valence', 'arousal', 'dominance']:
                wordShifts[m] = dict()

        for m in wordShifts:
            wordList = None
            wordProp = None
            if m in ['valence', 'arousal', 'dominance']:
                wordList = {w:self.sentiment.anewScores[w]['{}_mean'.format(m)] for w in self.sentiment.anewScores}
                wordPropR, wordPropC = sentiReference['anew_word_probability'], sentiComparison['anew_word_probability']
            if m == 'happiness':
                wordList = {w:self.sentiment.happinessScores[w]['{}_mean'.format(m)] for w in self.sentiment.happinessScores}
                wordPropR, wordPropC = sentiReference['happiness_word_probability'], sentiComparison['happiness_word_probability']

            # Keep analysis result in this dictionary
            sDiff = sentiComparison['{}_mean'.format(m)] - sentiReference['{}_mean'.format(m)]
            temp = {'difference': sDiff,
                    'reference': sentiReference['{}_mean'.format(m)],
                    'comparison': sentiComparison['{}_mean'.format(m)],
                    'word_shift': dict()}
            for w in set(wordPropC.keys()) | set(wordPropR.keys()):
                vDiff = wordList[w] - sentiReference['{}_mean'.format(m)]
                fDiff = wordPropC.get(w,0.) - wordPropR.get(w,0.)
                temp['word_shift'][w] = {'shift': (100. * vDiff * fDiff)/abs(sDiff),
                                         'case': '{}{}'.format('+' if vDiff > 0 else '-',
                                                               'u' if fDiff > 0 else 'd'),
                                         'value_difference': vDiff,
                                         'frequency_difference': fDiff,
                                         }
            wordShifts[m] = temp
        return wordShifts



class Sentiment(object):

    def __init__(self, config={}):
        self.config = config
        self.__path = os.path.dirname(os.path.abspath(__file__))
        self.happinessScores = self.__loadHappinessData()
        self.anewScores = self.__loadANEWData()

    def compute(self,text, methods=None, includeWordCounts=False, includeDistributions=False):
        if methods == None:
            methods = ['happiness', 'anew']
        senti = dict()
        if 'happiness' in methods:
            senti.update(self.happinessScore(text, includeWordCounts, includeDistributions))
        if 'anew' in methods:
            senti.update(self.anewScore(text, includeWordCounts, includeDistributions))
        return senti

    def _normPdf(self, xVal, meanV=0, stdV=1):
        return np.exp(-(xVal-meanV)**2/(2*stdV**2))/(np.sqrt(2*np.pi)*stdV)

    '''
    Peter Dodds's happiness words
    '''
    def happinessScore(self, text, includeWordCounts=False, includeDistributions=False):
        wordCounts = dict()
        for w in self.happinessScores:
            if not (self.config.get('happinessMin', 4.) < self.happinessScores[w]['happiness_mean'] < self.config.get('happinessMax', 6.)):
                wordCounts[w] = 0

        tScore = 0
        for w in text.lower().split():
            if w in wordCounts:
                wordCounts[w] += 1.

        xbins = np.linspace(0,10,101)
        sumWords = float(sum(wordCounts.values()))
        happiness = {'happiness_mean':0, 'happiness_std':0,
                     'happiness_distribution':[[d,0] for d in xbins]}
        if includeWordCounts:
            happiness['happiness_word_counts'] = {w:wordCounts[w] for w in wordCounts if wordCounts[w] <> 0}
            happiness['happiness_word_probability'] = {w:wordCounts[w]/sumWords for w in wordCounts if wordCounts[w] <> 0}

        for w in wordCounts:
            if wordCounts[w] == 0:
                continue
            happiness['happiness_mean'] += (wordCounts[w]/sumWords) * self.happinessScores[w]['happiness_mean']
            happiness['happiness_std'] += (wordCounts[w]/sumWords) * pow(self.happinessScores[w]['happiness_std'],2)

            if includeDistributions:
                for i,d in enumerate(xbins):
                    dpdf = self._normPdf(d,self.happinessScores[w]['happiness_mean'],
                                          self.happinessScores[w]['happiness_std'])

                    happiness['happiness_distribution'][i][1] += dpdf * (wordCounts[w] / sumWords)

        return happiness

    def anewScore(self, text, includeWordCounts=False, includeDistributions=False):
        wordCounts = dict()
        for w in self.anewScores:
            wordCounts[w] = 0

        for w in text.lower().split():
            if w in self.anewScores:
                wordCounts[w] += 1.

        xbins = np.linspace(0,10,101)
        sumWords = float(sum(wordCounts.values()))
        anew = {'valence_mean':0, 'valence_std':0, 'valence_distribution':[[d,0] for d in np.linspace(0,10,101)],
                'arousal_mean':0, 'arousal_std':0, 'arousal_distribution':[[d,0] for d in np.linspace(0,10,101)],
                'dominance_mean':0, 'dominance_std':0, 'dominance_distribution':[[d,0] for d in np.linspace(0,10,101)]}
        if includeWordCounts:
            anew['anew_word_counts'] = {w:wordCounts[w] for w in wordCounts if wordCounts[w] <> 0}
            anew['anew_word_probability'] = {w:wordCounts[w]/sumWords for w in wordCounts if wordCounts[w] <> 0}

        for w in wordCounts:
            if wordCounts[w] == 0:
                continue

            for t in ['valence', 'arousal', 'dominance']:
                anew['{}_mean'.format(t)] += (wordCounts[w]/sumWords) * self.anewScores[w]['{}_mean'.format(t)]
                anew['{}_std'.format(t)] += (wordCounts[w]/sumWords) * pow(self.anewScores[w]['{}_std'.format(t)],2)

                if includeDistributions:
                    for i,d in enumerate(xbins):
                        dpdf = self._normPdf(d,self.anewScores[w]['{}_mean'.format(t)],
                                              self.anewScores[w]['{}_std'.format(t)])

                        anew['{}_distribution'.format(t)][i][1] += dpdf * (wordCounts[w] / sumWords)
        return anew


    def __loadHappinessData(self):
        '''
        word,happiness_rank,happiness_mean,happiness_std,twitter_rank,google_rank,nyt_rank,lyrics_rank
        '''
        fname = '{}/data/hedenometer_data.txt'.format(self.__path)
        data = list()
        with open(fname) as f:
            for line in iter(f):
                data.append(line.split())

        sentiData = dict()
        headers = data[0][1:-1]
        for d in data[1:-1]:
            sentiData[d[0]] = dict()
            for i,h in enumerate(headers):
                try:
                    sentiData[d[0]][h] = float(d[i+1])
                except:
                    sentiData[d[0]][h] = d[i+1]
        return sentiData

    def __loadANEWData(self):
        '''
        word,wordid,valence_mean,valence_std,arousal_mean,arousal_std,dominance_mean,dominance_std,frequency
        '''
        fname = '{}/data/anew_word_values.txt'.format(self.__path)
        data = list()
        with open(fname) as f:
            for line in iter(f):
                data.append(line.split())
        anewData = dict()
        headers = data[0][1:-1]
        for d in data[1:-1]:
            anewData[d[0]] = dict()
            for i,h in enumerate(headers):
                try:
                    anewData[d[0]][h] = float(d[i+1])
                except:
                    anewData[d[0]][h] = d[i+1]
        return anewData




if __name__ == '__main__':
    testStr = 'War is a state of armed conflict between societies. It is generally characterized by extreme aggression, destruction, and mortality, using regular or irregular military forces. An absence of war is usually called "peace". Warfare refers to the common activities and characteristics of types of war, or of wars in general.[1] Total war is warfare that is not restricted to purely legitimate military targets, and can result in massive civilian or other non-combatant casualties. While some scholars see war as a universal and ancestral aspect of human nature,[2] others argue it is a result of specific socio-cultural or ecological circumstances.[3] The deadliest war in history, in terms of the cumulative number of deaths since its start, is the Second World War, from 1939 to 1945, with 60-85 million deaths, followed by the Mongol conquests[4] which was greater than 41 million. As concerns a belligerents losses in proportion to its prewar population, the most destructive war in modern history may have been the Paraguayan War (see Paraguayan War casualties). In 2013 war resulted in 31,000 deaths, down from 72,000 deaths in 1990.[5] In 2003, Richard Smalley identified war as the sixth (of ten) biggest problem facing humanity for the next fifty years.[6] War usually results in significant deterioration of infrastructure and the ecosystem, a decrease in social spending, famine, large-scale emigration from the war zone, and often the mistreatment of prisoners of war or civilians.[7][8][9] For instance, of the nine million people who were on the territory of Soviet Belarus in 1941, some 1.6 million were killed by the Germans in actions away from battlefields, including about 700,000 prisoners of war, 500,000 Jews, and 320,000 people counted as partisans (the vast majority of whom were unarmed civilians).[10] Another byproduct of some wars is the prevalence of propaganda by some or all parties in the conflict,[11] and increased revenues by weapons manufacturers.'
    testStr2 = 'Love is a variety of different feelings, states, and attitudes that ranges from interpersonal affection ("I love my mother") to pleasure ("I loved that meal"). It can refer to an emotion of a strong attraction and personal attachment.[1] It can also be a virtue representing human kindness, compassion, and affection-"the unselfish loyal and benevolent concern for the good of another".[2] It may also describe compassionate and affectionate actions towards other humans, ones self or animals.[3]Non-Western traditions have also distinguished variants or symbioses of these states; words like storge, philia, eros, and agape each describe a unique "concept" of love.[4] Love has additional religious or spiritual meaning-notably in Abrahamic religions. This diversity of uses and meanings combined with the complexity of the feelings involved makes love unusually difficult to consistently define, compared to other emotional states.Love in its various forms acts as a major facilitator of interpersonal relationships and, owing to its central psychological importance, is one of the most common themes in the creative arts.[5] Love may be understood as a function to keep human beings together against menaces and to facilitate the continuation of the species.[6]'
    #senti = Sentiment()
    #print senti.compute(testStr2, includeWordCounts=True)

    wShift = WordShifts()
    print wShift.computeWordShifts(testStr, testStr2)
