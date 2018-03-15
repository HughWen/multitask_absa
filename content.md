class: center, middle, no-number, titlepage
count: false



# Multi-task Learning for Aspect-level Sentiment Analysis

.author[Wen Weihuang]

.date[March 10, 2018]

---

# Outline

- What is aspect-level sentiment analysis
    - Sentiment analysis
    - Aspect-level sentiment analysis
- What is multi-task learning
- Why multi-task learning
- Recent Work
- Model description
- Experiments and analysis
- Conclusion and future work

---

# What is aspect-level sentiment analysis

- Sentiment analysis
- Aspect-level sentiment analysis

---

.subtitle[Sentiment analysis]

.font-7[📄 "Deep Learning for Sentiment Analysis: A Survey", Lei Zhang, Shuai Wang, Bing Liu, *arxiv.org*]

<q>Sentiment analysis or opinion mining is the computational study of people's opinions, sentiments, emotions, appraisals and attitudes towrads entities such as products, services, organizations, individuals, issues, events, topics, and their attributes.</q>

<q>Since early 2000, sentiment analysis has grown to be one of the most active research areas in natural language processing (NLP). It is also widely studied in data mining, Web mining, text mining, and information retrieval. </q>

Researchers have mainly studied sentiment analysis at three levels of granularity:
- document level
- sentence level
- .alert[**aspect level**]

---

.subtitle[Aspect-level sentiment analysis]

<q>Compared with document level and sentence level sentiment analysis, aspect level sentiment analysis or aspect-based sentiment analysis is more fine-grained. Its task is to extract and summarize people's opinions expressed on entities and aspects/features of entities, which are also called targets.</q>

<q>For example, in a product review, it aims to summarize positive and negative opinions on different aspects of the product respectively, although the general sentiment on the product could be positive or negative. The whole task of aspect-based sentiment analysis consists of several subtasks such as .alert[**aspect extraction**], .alert[**entity extraction**] and .alert[**aspect sentiment classification**].</q>

---
.subtitle[Aspect-level sentiment analysis]

.dl[Demo: "the voice quality of iPhone is great, but its battery sucks".

- entity extraction:
    -  "iPhone"
- aspect extraction:
    -  "voice quality"
    -  "battery"
- aspect sentiment classification: 
    - "voice quality": positive
    - "battery": negative
]

---

# What is multi-task learning

Todo

---

# Why multi-task learning

Todo

---

# Recent work

- Artificial neural networks
- Word embedding
    - Word2Vec
        - Continuous Bag-of-Words
        - Skip-Gram
    - Global Vector
- Autoencoder and denoising autoencoder
- Convolutional neural network
- Recurrent neural network (RNN)
    - Long short-term memory network (LSTM)
    - Bidirectional RNN
    - Gated Recurrent Unit (GRU)

---

# Recent work
- Attention mechanism with recurrent neural network
- Memory network
- Recursive neural network


---

# Model description

Todo

---

# Experiments and analysis

Todo

---

# Conclusion and future work

Todo
