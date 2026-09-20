---
description: Run or extend the exploratory data analysis on the Disneyland reviews corpus
---

Act as a data scientist specialising in exploratory data analysis, NLP and sentiment
analysis.

Produce a complete, reproducible EDA of the Disneyland reviews corpus before any model is
trained, for **Team 4 — Sentiment Analysis (NLP)**.

## The dataset

Loaded with `from ml_npl.dataset import load_reviews`. Kaggle handle
`arushchillar/disneyland-reviews`, 42,656 rows, six columns, **zero nulls**:

| Column | Role | Notes |
| --- | --- | --- |
| `Review_ID` | identifier | Per **review**, not per user. 42,636 distinct in 42,656 rows. |
| `Rating` | ordinal 1-5 | Source of the sentiment target. Never a feature. |
| `Year_Month` | period `"YYYY-M"` | Month granularity. Encodes absent dates as the literal string `"missing"`. |
| `Reviewer_Location` | country | 162 levels, 34% United States. |
| `Review_Text` | free text | The only legitimate model input. |
| `Branch` | park | 3 levels: California 45.5%, Paris 32.0%, Hong Kong 22.6%. |

Verify this schema on load rather than trusting it. If it has changed, say so and adapt.

## Three limits of this corpus

These are settled. Do not spend steps rediscovering them, and do not silently skip the
analyses they block — state the limitation instead.

1. **No user column exists.** Per-user analysis, repeat-reviewer bias and group-split-by-user
   are all impossible. Any result from this corpus carries that as a stated limitation.
2. **No day component.** `Year_Month` stops at the month, so there is no day-of-week
   analysis. Temporal work is monthly.
3. **The corpus is monolingual.** A 3,000-review sample detected 100% English. Treat the
   language section as a verification, not an exploration.

## Already established — build on these, do not re-derive

* Rating distribution is 1:3.5% 2:5.0% 3:12.0% 4:25.3% 5:54.3%. Majority-class accuracy
  is **79.5%**, which disqualifies accuracy as a headline metric.
* Quality defects: 12 fully duplicated rows, 24 duplicated review texts, 20 duplicated
  `Review_ID`s, 2,613 `"missing"` sentinels, 172 reviews over 1,000 words, 59 with URLs.
  Zero empty, zero emoji, zero out-of-range ratings.
* `Review_ID` correlates **+0.9927** (Spearman) with the visit month — it is a timestamp
  in disguise. Exclude it as a feature; use it to order the 2,613 undated rows.
* `n_words` and `n_chars` correlate **1.00**. They are one feature, not two.
* `Branch` confounds the label: 13.7% of Paris reviews are negative against 6.4% of
  California's.
* Top n-grams are topic terms (`fast pass`, `space mountain`), not sentiment terms.

## Ground rules

* Inspect the real structure first. Never assume a column that does not exist.
* Back every conclusion with a statistic or a figure. Invent nothing.
* **Delete nothing.** Quantify each problem and propose a treatment; the caller decides.
* Distinguish data-quality defects from natural characteristics of the corpus.
* Never overwrite `Review_Text` or any original column. Derived columns are additive.
* Keep exploration separate from training preprocessing. Nothing learned from the test
  set may inform cleaning, vocabulary, balancing, feature selection or hyperparameters.

## Sections

**1. Initial inspection.** Shape, dtypes, head and tail, descriptive statistics for
numeric and categorical columns, null percentages, duplicate counts, cardinality,
constant or near-constant columns, candidate unique keys. Summarise as a table of
column / dtype / unique / nulls / %null / interpretation.

**2. Data quality.** Missing values, duplicate rows, duplicate reviews, empty reviews,
extremely short and extremely long reviews, inconsistent values, formatting errors,
special characters, encoding problems, ratings out of range, invalid periods,
inconsistent category values. Count each and propose a treatment. Remove nothing.

**3. Target analysis.** Frequency (absolute and relative), mean, median, mode, standard
deviation, quartiles. Plot the distribution. Then evaluate — do not assume — how the
rating becomes a sentiment label. Present at least the three-class scheme (1-2 / 3 / 4-5),
the binary scheme (1-3 / 4-5) and the drop-neutral option, with the class counts and the
majority-to-minority ratio for each, and argue for one. State whether each class has
enough examples for classical ML and for deep learning, and what the imbalance implies
for training.

**4. Review analysis.** Per review: characters, words, sentences, unique words, type/token
ratio. Distribution via histogram and boxplot. Flag very short and very long reviews,
possible spam, repeated reviews, symbol-only reviews, heavy capitalisation, heavy
exclamation use, URLs and emoji — with counts, not removals.

**5. Linguistic analysis.** Total and unique word counts, top terms overall, per rating
and per sentiment class. Keep the original text and work on a normalised copy. Explain
which transformations suit bag-of-words models (Naive Bayes, logistic regression, SVM)
and which are counterproductive for RNNs, LSTMs and transformers. Call out the stopword
trap explicitly: NLTK's English list contains `not`, `no` and `nor`, and removing them
inverts the sentiment the task exists to detect.

**6. N-grams.** Unigrams, bigrams and trigrams for the whole corpus and per sentiment
class. Identify expressions characteristic of each class through the contrast *between*
classes. Frequency alone does not make a term positive or negative.

**7. Length vs rating.** Rating against words, characters and sentences. Report rank
correlations. Describe associations; claim no causality.

**8. Analysis by park.** Counts, mean and median rating, rating distribution, sentiment
mix and median length per park. Describe differences; do not rank the parks as better or
worse, and note that reviewers self-select.

**9. Temporal analysis.** Parse `Year_Month`, turning `"missing"` into `NaT`. Extract year,
month and year-month — there is no day. Review volume, mean rating, sentiment mix and
length over time. Handle thin periods explicitly rather than plotting noisy averages, and
note that the first and last years are partial. Never put volume and rating on twin
y-axes; use two charts.

**10. Language check.** Confirm the corpus is monolingual on a sample. If non-English
reviews appear, count them and discuss removal, translation or separate treatment — do
not remove them.

**11. Outliers.** IQR on the length metrics. Separate statistical extremes from genuinely
long or short reviews and from data errors. Note that an ordinal five-level rating has no
meaningful IQR outliers. Do not remove anything merely for being extreme.

**12. Data leakage — mandatory.** For every column, state whether it is a feature, target
only, or excluded, and why. Cover at minimum: `Rating` and anything derived from it,
`Review_ID`, `Year_Month`, `Branch`, `Reviewer_Location`, and the derived length features.
The rating that builds the label must never be a model feature. Also check feature
redundancy between the derived metrics.

**13. Bias.** Class imbalance, differences between parks, temporal drift, differences by
country, duplicated reviews, very long reviews. Explain how each could affect the model.
Include the absence of a user column as an unassessable bias, not an absent one.

**14. Visualisations.** At minimum: rating distribution, sentiment class balance, review
length distribution, length by rating, sentiment mix by park, mean rating by park, review
volume over time, mean rating over time, top terms, top bigrams, top terms by sentiment,
and a correlation matrix over the numeric features. Every figure needs a title, labelled
axes, units where they apply, and a legend whenever more than one series is present.

Assign colour by the job it does: a **sequential** ramp for the ordered rating, a
**diverging** scale with a neutral midpoint for sentiment polarity and for correlations,
**categorical** hues for the parks. Validate any categorical palette for colour-vision
deficiency rather than eyeballing it, and give low-contrast fills direct value labels.
Draw no figure that adds nothing.

**15. Preprocessing plan.** Separate the columns to preserve untouched from the columns to
derive. Give the pipeline as an ordered table, and for each step state whether it is fit
on the whole dataset or on the training split only. Anything learned — vocabularies,
vectorisers, resampling, thresholds — is training-only.

**16. Train / validation / test.** Propose a strategy and argue it. Evaluate random,
stratified and temporal splits on their merits; state that a group split by user is
impossible here. Deduplicate review texts **before** splitting, or identical texts land on
both sides and inflate the score invisibly.

**17. Conclusions.** Main findings backed by numbers. Quality problems with their
magnitude. Target distribution and imbalance. Text characteristics. Risks for modelling —
leakage, imbalance, duplicates, temporal drift, confounding, and the unassessable user
dimension. Concrete preprocessing recommendations. Then an experimental plan: majority
baseline, TF-IDF with logistic regression, TF-IDF with SVM and Naive Bayes, LSTM,
fine-tuned transformer. Name the metric and justify it against the class balance.

## Implementation

Python with pandas, numpy, matplotlib, seaborn, nltk and scikit-learn — all already
installed. Reusable logic belongs in `ml_npl/eda.py` and `ml_npl/plots.py`, not inline in
the notebook; the later workshops reuse those definitions. The notebook
(`notebooks/EDA.ipynb`) is organised in the sections above, one concern per cell, never
one cell for everything.

Code, identifiers, docstrings and notebook prose are in English, matching the assignment.

Fix `RANDOM_STATE = 42` and keep it constant: Workshop #2 is benchmarked against the
Workshop #1 numbers, so the split, the seed and the metric must not drift.

Execute the notebook end to end before calling it done, and confirm that every cell ran
without error. A notebook that has not been run is not a deliverable.

## Deliver

The executed EDA, its figures, the quality table, the target analysis, the text analysis,
the leakage verdicts, the preprocessing recommendations, the split strategy, the
conclusions, and the ordered next steps for training.
