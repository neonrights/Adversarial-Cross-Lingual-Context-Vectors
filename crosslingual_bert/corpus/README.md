# Data Generation Classes

## CorpusReader

Parent class for corpus readers.  Inherit from this class and implement extract sentences to produce a new reader for a specific corpus.

extract_sentences(document_name): returns a dictionary containing a sequence of sentences sampled from the specified in-corpus document.  Dictionary can also contain translation or text-alignment if needed.

## EnCzWordReader

Corpus reader for the Czech-English Manual Word Alignment corpus. (download)[https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804]

Please refer to this reader when implementing your own.

## LanguageSequenceGenerator

Samples sequences of sentences from a corpus for a specific language.  Outputs sequences as jsonl file containing sequence of sentences.  If available in the corpus, it also outputs parallel sentences and text alignment.

## DiscriminatorSequenceGenerator

Samples sequences of sentences from a set of corpora creating a jsonl file with each line containing the sequence of sentences and the language it's from.