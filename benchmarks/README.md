# Benchmarks

`raw/` holds the files of the seven standard benchmarks as their authors released them. Only the files that
`muster/benchmarks.py` reads are included, unchanged (md5 in `MD5SUMS`). `encodings/` holds the frozen encodings the
paper's tables were computed from, with the paper id of every row in `encodings/ids/`.

| Benchmark | Source | Scored items | What is compared |
|---|---|---|---|
| LR-PC, LR-RC | LR-Bench, released with RATE (Liu et al., ACL 2026) | 897 (paper, author) items | 212 and 1,184 pairwise comparisons of graded expertise |
| CMU Gold | Stelmakh et al., A Gold Standard Dataset for the Reviewer Assignment Problem (arXiv:2303.16750) | 477 (paper, researcher) items, 10 profile samples | 1,653 comparisons of self-reported expertise |
| NIPS | Mimno and McCallum, KDD 2007 | 393 judgments, 34 queries, 190 reviewers | ranking of the judged reviewers per query |
| SIGIR | Karimzadehgan, Zhai and Belford, CIKM 2008 | 13,797 judgments, 73 queries, 189 reviewers | ranking of the judged reviewers per query |
| SciRepEval | Singh et al., EMNLP 2023 | 1,729 judgments, 107 queries, 661 reviewers | ranking of the judged reviewers per query |
| KDD | Karimzadehgan and Zhai, CIKM 2009, and Zhang et al., WWW 2025 | 3,480 judgments, 174 queries, 737 reviewers | ranking of the judged reviewers per query |

The four ranking benchmarks are used in the form packaged by Chain-of-Factors (Zhang et al., WWW 2025), with soft and hard
relevance labels. The CMU Gold Standard files come from github.com/niharshah/goldstandard-reviewer-paper-match, whose
`LICENSE-DATA.md` sets the data terms. The LR-Bench files include their dataset card (`raw/lr_bench/README.md`). Check each
source's terms before redistributing these files.

## Encodings

`encodings/<enc>/{lr,cmu,NIPS,SIGIR,SciRepEval,KDD}.npy` are float32 unit vectors, one row per paper id in
`encodings/ids/`. These are the matrices the paper's MUSTER rows were scored with (docs/DATA.md describes how each was
encoded). `scripts/encode.py benchmarks` re-encodes them, and `scripts/encode.py check` compares a sample with these files.

## Metrics

- **Preference:** the accuracy of a track is the share of comparisons ordered correctly. A tie counts as wrong. The loss
  is the share ordered wrongly, weighted by the gap in the human labels, with a tie counting half.
  - CMU reports 100 times one minus its loss, averaged over the 10 profile samples, with the unweighted accuracy in
    brackets.
  - The macro averages the three tracks' rounded values.
- **Ranking:** each query's judged reviewers are ranked by score, with ties broken by the reviewer id as a string.
  - P@k is the number of relevant reviewers in the top k divided by k, with soft labels (y >= 1) or hard labels (y >= 2).
  - nDCG@3 uses the gain 2^y - 1.
  - A query without a relevant reviewer counts as 0.
  - The macro averages the four benchmarks.
