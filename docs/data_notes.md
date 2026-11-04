# Data notes

| Item                                  | Value     |
| ------------------------------------- | --------- |
| Wikipedia pages indexed               | 5,396,106 |
| Sentences indexed                     | 25,247,887 |
| wiki.db size                          | 9.06 GB (5.4 min build) |
| Train / val / test claims             | 145,449 in train.jsonl (val is 5% of it) / 19,998 test |
| Label distribution (train)            | SUPPORTS 55%, NEI 25%, REFUTES 20% |
| Gold sentences missing from dump      | 0 (page ids normalized to NFC) |
| NEI strategy used for headline models | retrieved |

Two quirks in the dump, both handled when building wiki.db:

- Claim files write accents as separate marks (NFD) but the dump uses NFC, so "Björk" did not
  match. Evidence page ids are normalized to NFC when loading claims (72 dev and 286 train gold
  sentences were affected).
- One page (Duluth_Kelleys/Eskimos) has link text on its own line that looks like a sentence
  numbered 1927. The builder keeps only lines numbered 0, 1, 2, ... in order and reports
  how many it dropped: 3, all on that page, so the dump has no real gaps in numbering.
