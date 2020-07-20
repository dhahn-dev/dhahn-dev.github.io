---
title: Word Piece Model
tags: nlp, BERT
---

토크나이징은(tokenizing): 문장을 토큰으로 나누는 과정

텍스트 데이터를 학습한 모델의 크기는 단어의 개수에 영향을 받음

RNN 기반의 알고리즘들은 단어 개수에 비례하여 계산 비용이 증가

그렇다고 vocabulary의 개수를 제한하게 되면 임베딩 벡터로 표현하지 못하는단어가 생김(OOV)

RNN 기반 모델에서 이를 해결하기 위해서 Word Piece Model(WPM)이 제안됨

단어를 한정적인 유닛(fninite subword units)으로 표현

WPM은 언어에 상관없이 모두 적용할 수 있기 때문에 적용할 언어마다 해당 언어의 특징을 반영한 토크나이저를 만들지 않아도 됨

하지만 모든 데이터 분석에 적합한 것은 아님

## **Word Piece, units of words**

WPM은 제한적인 vocabulary units, 정확히는 단어를 표현할 수 있는 subowrds units으로 모든 단어를 표현함

BOW의 경우 수 백만개의 단어를 포함하는 데이터를 표현하기 위해서 단어 개수만큼의 차원을 지닌 벡터 공간을 이용

RNN과 같이 word embedding vectors를 이용하는 모델은 단어 개수 만큼의 embedding voector를 학습하기 때문에 단어의 개수가 많을수록 차원이 커지고 모델이 무거워짐

따라서 제한된(finite)개수의 단어를 이용하는 것이 필요

하지만 단순히 자주 이용되지 않는 단어들을 제외하게 되면 OOV 문제가 발생

언어는 글자(characters)를 subword units으로 이용함

* 영어(alphabet): 대부분의 영어 단어는 몇개의 글자가 모여 하나의 단어를 구성

    하나의 유닛이 어떤 개념을 지칭하기 어렵다
    
* 중국어: 여러 글자가 모여 하나의 단어를 이루기도 하지만, 한 글자로 구성된 단어도 많음
    
    영어보다는 유닛의 모호성이 줄어듬
    
    동음이의어의 문제가 있지만, 가장 모호성이 적은 방법은 모든 단어를 유닛으로 이용하는 것

토크나이징 방법에 따라 모호성이 적은 최소한의 유닛을 만들 수도 있음

하지만 문제는 언어에 맞는 최적의 토크나이징을 하려면 해당 언어의 언어학적 지식과 학습데이터가 필요함

언어가 다르고, 도메인이 다르면 이를 준비하는 것은 어려움

## **Word Piece Model (sentencepiece) tokenizer**

학습 데이터를 이용하지 않으면서도 모호성이 적은 최소한의 unit을 만드는 heuristics이 존재

BPE와는 달리 빈도수가 아니라 우도(likelihood)를 통해서 단어를 분리

유닛인 subwords의 경우 유닛이 아닌 subwords보다 자주 등장할 가능성이 높다는 특징을 이용

유닛이 자주 등장한 다는 것은 아마도 많은 언어의 공통적인 특징일 것

이를 이용하면 language independent, universial tokenizer를 만들 수 있을 것

WPM도 위의 개념을 이용한 토크나이저


WPM은 모든 단어의 시작에  underbar를 붙이는데, 이는 문장 생성, 혹은 subwords로부터의 문장 복원을 위한 특수기호임

underbar없이 subwords를 띄어두면 본래 띄어쓰기와 구분이 되지 않기 떄문

WPM에서는 자주 등장하는 words 자체를 units으로 이용하고, 자주 등장하는 단어는 subword units으로 나눔

문장을 복원하기 위해서는 띄어쓰기를 기준으로 나눠진 token들을 concatenation한 후 underbar를 기준으로 다시 tokenizing하거나 빈 칸으로 치환하여 문장을 복원

```python
def recover(tokens):
    sent = ''.join(tokens)
    sent = sent.replace('_', ' ')
    return sent
```

## **Byte-pair Encoding**

BPE(Byte pair Encoding) 알고리즘은 1994년에 제안된 데이터 압축 알고리즘으로, 자연어 처리에서 단어 분리 알고리즘으로 응용됨

BPE는 기본적으로 연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합하는 방식을 수행함

자연어 처리에서의 BPE는 단어 분리(word segmentation) 알고리즘으로 기존에 있떤 단어를 분리한다는 의미

글자(character) 단위에서 점차적으로 단어 집합(vocabulary)를 만들어 내는 Bottom up 방식의 접근을 사용
훈련 데이터에 있는 단어들을 모든 글자(characters) 또는 유니코드(unicode) 단위로 단어 집합(vocabulary)를 만들고, 가장 많이 등장하는 유니그램을 하나의 유니그램으로 통합

아래의 vocab은 low,lower,newest,widest의 맨 뒤에 특수기호 '/w'를 넣은 뒤, 한글자 단위로 모두 띄어 초기화를 한 상태

Character는 기본 subword units이며, for loop에서 빈도수가 가장 많은 bigram을 찾음

선정된 bigram은 하나의 unit으로 merge하고, 이 과정을 num_merges만큼 반복

vocab의 value는 빈도수로 'low'가 5번, 'lower'가 2번 등장

>
```python
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>' : 5,
         'l o w e r </w>' : 2,
         'n e w e s t </w>':6,
         'w i d e s t </w>':3
         }

num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)

print(vocab)
```


토크나이저 입장에서는 많이 쓰이는 subwords를 units으로 이용하면 자주 이용되는 단어는 그 자체가 unit이 되며, 자주 등장하지 않는 단어(rare words)가 subword units으로 나뉘어짐

즉, WPM은 각 언어의 지식이 없이도 빈번히 등장하는 substring을 단어로 학습하고, 자주 등장하지 않는 단어들을 최대한 의미보존을 할 수 있는 최소한의 units으로 표현

한국어에서 복합명사를 단일명사들로, 어절을 명사와 조사로 나누는 것과 비슷함



---






문서분류에는  bigram이 유용하다!



