# AI Assignment

## Q1. How does a real search engine work? Describe important NLP tasks and the deep-learning algorithms used for them.

A web search engine is an information-retrieval system. Its purpose is to return useful, trustworthy results from a very large collection of documents in a fraction of a second. Although the exact design differs between companies, most modern search engines follow four connected stages.

### 1. Crawling

Web crawlers, sometimes called spiders, discover pages by following hyperlinks, reading sitemaps, and revisiting pages that may have changed. Before fetching a page, a crawler should follow the site's `robots.txt` rules and manage its request rate so that it does not overload the site.

### 2. Processing and indexing

After downloading a page, the engine extracts useful content such as the title, headings, body text, links, images, and structured data. It removes boilerplate where possible and stores terms in an **inverted index**. An inverted index maps each term to the documents in which it occurs; therefore, the engine does not need to scan every web page for every query.

### 3. Retrieval

When a user submits a query, the engine first finds a manageable set of candidate documents. Traditional lexical retrieval methods such as **TF-IDF** and **BM25** are still valuable because they quickly find documents containing important query terms. Modern systems often add **dense retrieval**, where the query and documents are represented as vectors and nearest-neighbour search retrieves documents with related meanings.

### 4. Ranking and presentation

Candidate documents are ranked using many signals: textual relevance, freshness, quality, language, location when appropriate, and signals intended to reduce spam. A final ranking model may re-order the best candidates. The engine then generates a results page with titles, URLs, snippets, and sometimes direct answers or knowledge panels.

### Important NLP tasks

- **Tokenisation and normalisation:** Text is split into words or subwords. Case-folding, stemming, or lemmatisation may group related forms such as *diagnose*, *diagnosed*, and *diagnosis*.
- **Spelling correction and query understanding:** The engine can identify likely typing errors, segment queries such as `bestlaptop2026`, and interpret abbreviations or synonyms.
- **Named Entity Recognition (NER) and entity linking:** The system identifies entities such as people, organisations, places, products, and diseases. Entity linking distinguishes, for example, *Jaguar* the animal from the car brand.
- **Intent classification:** Queries may be informational ("symptoms of leaf blight"), navigational ("IRCTC login"), transactional ("buy soil moisture sensor"), or local. Intent helps the system choose suitable results and features.
- **Semantic matching:** The engine estimates whether a document answers the meaning of a query even when it uses different words. For example, a page about "treatment for fungal spots on tomato leaves" may be relevant to "how to cure tomato leaf fungus."
- **Passage retrieval and summarisation:** Rather than treating a long page as one unit, a system may identify the most relevant passage and generate a short snippet around it.

### Deep-learning methods used in search

| Method | Typical role in a search engine |
|---|---|
| **Word and subword embeddings** (Word2Vec, GloVe, FastText) | Represent words numerically so that related terms have nearby representations. FastText is especially useful for rare or misspelled words because it uses subword information. |
| **Transformer encoders** (BERT and related models) | Build context-sensitive representations. They help with query understanding, NER, semantic matching, and ranking because a word's meaning depends on its surrounding words. |
| **Bi-encoders / dual encoders** | Encode a query and each document independently into vectors. This makes dense retrieval practical at large scale because document vectors can be precomputed. |
| **Cross-encoders** | Read a query and a candidate passage together, giving a more accurate relevance score. They are usually applied only to a small candidate set because they are computationally expensive. |
| **Learning-to-rank neural models** | Learn to combine relevance features from labelled examples or user feedback. These models can improve the ordering of retrieved results. |

In practice, search engines use a hybrid approach. Fast lexical retrieval provides speed and precise keyword matching, while neural models improve understanding and ranking. Deep learning therefore complements—not completely replaces—classical information-retrieval methods.

## Q2. Discuss the NLP concepts behind plagiarism checking (similarity checking) and AI-writing detection in platforms such as Turnitin and QuillBot. Why can paraphrasing affect the results?

Similarity checking and AI-writing detection are different tasks. A similarity checker looks for overlap between a submission and material in a reference collection. An AI-writing detector estimates whether a passage has statistical characteristics associated with machine-generated text. Neither score, by itself, is a final academic-integrity judgement.

### 1. NLP concepts behind similarity checking

Similarity systems compare a submitted document against available web pages, publications, and, where permitted, prior student submissions. Their main steps include the following.

- **Text preparation:** The document is divided into tokens and sentences. Quotations, references, and very common phrases may be handled separately so that they do not dominate the result.
- **N-grams and phrase matching:** An *n*-gram is a sequence of *n* consecutive tokens. Comparing word n-grams helps find copied or closely edited phrases.
- **Fingerprinting and hashing:** For efficient search, a system can select representative word sequences and convert them into compact fingerprints. Techniques such as winnowing allow a long document to be compared with a large database without storing every possible phrase.
- **Candidate retrieval:** An inverted index or similar structure quickly identifies documents sharing phrases with the submission. More detailed comparison is then done only for those candidates.
- **Similarity scoring and source alignment:** The system highlights matching spans and reports their sources. It may use exact matching, approximate string matching, or vector-based semantic similarity to identify lightly modified wording.

The reported similarity percentage is not the same as a plagiarism verdict. Correctly quoted text, a bibliography, standard technical terms, and properly cited evidence can all create matches. Turnitin states that the instructor must interpret the report and make the academic judgement.

### 2. NLP concepts behind AI-writing detection

The exact methods used by commercial products are proprietary, so they should not be described as a single universal algorithm. In general, an AI detector may use a classifier trained on examples of human and machine-generated writing. Possible features include:

- **Token-probability patterns:** Language models assign probabilities to possible next tokens. Measures related to predictability, including perplexity, may be used as one signal. Low perplexity alone does not prove AI authorship because careful human academic writing can also be predictable.
- **Stylometric features:** Sentence length, punctuation, vocabulary distribution, repetition, transitions, and syntactic patterns can be represented as numerical features. The informal term *burstiness* refers to variation in these features, but it is not a reliable proof of authorship.
- **Embedding-based classifiers:** Transformer models can encode a passage into contextual vectors. A classifier can use these representations to estimate how similar the passage is to the examples on which it was trained.
- **Segment-level analysis:** A system may score sections of a document rather than treating the whole document as one block. This can identify passages that have different writing characteristics.

These methods face an important limitation: human and AI writing can overlap greatly, especially for short, formulaic, heavily edited, or non-native-English text. As a result, false positives and false negatives are possible. Detection output should be treated as evidence for review, not as conclusive proof.

### 3. Why paraphrasing can affect results

Paraphrasing changes surface wording, sentence structure, and word frequency. Therefore, it can reduce **exact phrase overlap** and may change the statistical signals used by an AI classifier. A similarity checker may find fewer direct n-gram matches after paraphrasing, while an AI detector may return a different probability because the text's style has changed.

However, paraphrasing is not a dependable way to determine authorship or avoid academic review. Systems can still identify matching ideas, uncommon phrases, citations, or source structure, and AI-detection results can change simply because the models are probabilistic and imperfect. More importantly, changing wording without crediting the original source can still be plagiarism. The appropriate academic practice is to write from one's own understanding, cite every borrowed idea or quotation, retain notes and drafts, and follow the course policy on AI assistance.
