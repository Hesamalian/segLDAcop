# Topical Coherence in LDA-based Models through Induced Segmentation

To run the code, it needs to have toy_dataset.text as an example and vocabulary.py for pre-processing in the same path,


You can run the code simply by:

python segLDAcop.py [path2train] [Number of topics] [Number of iterations] [Removing stop words]

For instance:

python segLDAcop.py toy_dataset.txt 20 50 false

Output:

perp.txt : Perplexity at each iteration

boundtopicslast : Segmented topic assigments for the whole corpus

please use this bibtex  to cite this work: 

@InProceedings{P17-1165,
author =     "Amoualian, Hesam
and Lu, Wei
and Gaussier, Eric
and Balikas, Georgios
and Amini, Massih R
and Clausel, Marianne",
title =     "Topical Coherence in LDA-based Models through Induced Segmentation",
booktitle =     "Proceedings of the 55th Annual Meeting of the Association for      Computational Linguistics (Volume 1: Long Papers)    ",
year =     "2017",
publisher =     "Association for Computational Linguistics",
pages =     "1799--1809",
location =     "Vancouver, Canada",
doi =     "10.18653/v1/P17-1165",
url =     "http://www.aclweb.org/anthology/P17-1165"
}
