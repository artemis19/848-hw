# Notes

> Group 10, Team SKY | Friday, February 11, 2022
-------------------------------------------------

## Best Performances

Turning on Kaitlyn's features & year vector using the `feat_utils_backup.py`:

```
accuracy            :  14.831
expected_win_prob   :   0.147
buzz_percent        :  40.853
mean_buzz_position  :   1.200
```

Using my features alone in the training stage as well using `feat_utils.py`:

```
accuracy            :  25.488
expected_win_prob   :   0.163
buzz_percent        :  99.911
mean_buzz_position  :   0.433
```

Using OG features and guess occurrence feature:

```
accuracy            :  25.755
expected_win_prob   :   0.178
buzz_percent        :  97.691
mean_buzz_position  :   0.495
```

W/o question disambiguation and guess occurrence feature:

```
accuracy            :  25.755
expected_win_prob   :   0.177
buzz_percent        :  97.780
mean_buzz_position  :   0.492
```

## Quickstart

I did everything in a virtual environment. I think you'll need at least `python3.8` for everything to work correctly. I did not upload my two JSON output files from modifying the couple of features, so you'll need to run the `tfidf_guesser` to get those initial ones and see. I have a `.gitignore` for large or python compiled files but feel free to add anything else that comes up.

## Generating & Training Model

Modify features in `featng/feat_utils.py`

Then run the guesser and save the outputs:

```python
python3 tfidf_guesser.py --buzztrain_predictions train.pred --buzzdev_predictions dev.pred
```

`buzztrain` is the training data for guesses while the `buzzdev` is the validation data.

Then pass the buzzer your output files:

```python
python3 lr_buzzer.py --buzztrain train.pred --buzzdev dev.pred
```
# Baseline - 2/4 Repo Status

## Baseline Metrics

With no features added, this was the initial baseline with the small dataset:

```
accuracy            :  53.879
expected_win_prob   :   0.445
buzz_percent        :  90.086
mean_buzz_position  :   0.742
```

## Features Added

### Feature - How many space separated words have we encountered?

* Example one from README
	* input `question_text`

Performance metrics:

Buzzer metrics:

```
Buzzer Train Accuracy: 0.75
Buzzer Dev Accuracy  : 0.61
```

Evaluation metrics:

```
accuracy            :  45.690
expected_win_prob   :   0.302
buzz_percent        : 100.000
mean_buzz_position  :   0.289
```

### Feature - Using the `run_length` alone - DID NOT IMPROVE

* Added a feature that would attempt to use how far into the question the model could buzz & guess
	* Attempts to use the position in the question to choose its guess correctly

Buzzer metrics:

```
Buzzer Train Accuracy: 0.75
Buzzer Dev Accuracy  : 0.61
```

Evaluation metrics:

```
accuracy            :  45.690
expected_win_prob   :   0.302
buzz_percent        : 100.000
mean_buzz_position  :   0.289
```

__Removing the `question_text` feature__

`question_text` seemed to decrease the original baseline metrics, so I removed this while keeping the `run_length` feature. It only improved by a small amount.

```
accuracy            :  45.690
expected_win_prob   :   0.325
buzz_percent        : 100.000
mean_buzz_position  :   0.306
```

# Baseline - 2/7 Repo Status

```
accuracy            :   6.306
expected_win_prob   :   0.056
buzz_percent        :  91.119
mean_buzz_position  :   0.036
```

__This now incorporates the qanta large datasets, so everything is lower to start.__

## Features Added

List of potential features:

* run_length
* score
* log2(question_text)
* category average
* disambiguation in the question
* guess aggregation average

### `run_length`

Including `scores` and `run_length`:

```
accuracy            :   7.815
expected_win_prob   :   0.051
buzz_percent        : 100.000
mean_buzz_position  :   0.054
```

More data from the question should give you a higher chance of getting the question correct, so choosing a guess with the highest "run length," or position in the question should give us a better accuracy in choosing a correct guess.

### Question Disambiguation

Including `scores`, `run_length`, and `disambiguation`:

```
accuracy            :   8.171
expected_win_prob   :   0.053
buzz_percent        : 100.000
mean_buzz_position  :   0.062
```

For a question that has a disambiguation in it, extract the word up until the last letter and then count up how many times it appears in the question text.

Extract up until the last letter because a word such as "philosophy" without the "y" could match to similar words like "philosopher" or "philosophical."

### Category

Including `scores`, `run_length`, `disambiguation`, and `category`:

```
accuracy            :   8.171
expected_win_prob   :   0.054
buzz_percent        : 100.000
mean_buzz_position  :   0.063
```

If the category text is mentioned in the question, it increases a count metric increasing the chances that that is the correct guess.

### Sub-category

```python
python tfidf_guesser.py --guesstrain ../data/qanta.train.json --guessdev ../data/qanta.dev.json --buzztrain_predictions train.pred --buzzdev_predictions dev.pred
```

Including `scores`, `run_length`, `disambiguation`, `category`, and `sub_category` trained on the larger qanta datasets:

```
accuracy            :  11.723
expected_win_prob   :   0.076
buzz_percent        : 100.000
mean_buzz_position  :   0.202
```

If the subcategory "root of the word" text is mentioned in the question, it increases a count metric increasing the chances that that is the correct guess.


## Testing Features - Small Dataset

Features that give the logistic regression convergence warning:

* tournament
* page

### Training on small data without vectorized features:

```
accuracy            :   7.815
expected_win_prob   :   0.051
buzz_percent        : 100.000
mean_buzz_position  :   0.054
```

Adding in `difficulty`:

```
accuracy            :   7.371
expected_win_prob   :   0.055
buzz_percent        : 100.000
mean_buzz_position  :   0.055
```

Adding in `year`:

```
accuracy            :  12.256
expected_win_prob   :   0.098
buzz_percent        :  65.808
mean_buzz_position  :   0.421
```

Added `difficulty` & `year`:
```
accuracy            :  11.901
expected_win_prob   :   0.083
buzz_percent        :  89.343
mean_buzz_position  :   0.227
```

Added `subcategory`, `difficulty` & `year`:

```
accuracy            :   9.147
expected_win_prob   :   0.060
buzz_percent        : 100.000
mean_buzz_position  :   0.091
```

Added `category`, `subcategory`, `difficulty` & `year`:

```
accuracy            :   8.437
expected_win_prob   :   0.057
buzz_percent        :  98.401
mean_buzz_position  :   0.080
```

### Testing w/Just Vectorized Features

Added `diffculty` & `year`:

```
accuracy            :  11.901
expected_win_prob   :   0.083
buzz_percent        :  88.632
mean_buzz_position  :   0.229
```

Adding in `disambiguation`, `category_points`, or `sub_category` did not affect the above metrics.

Using only the `year` vectorized feature seemd to improve metrics the best and adding in the other features also improved expected win probability.

`year` with `disambiguation` and `category_points`:

```
accuracy            :  12.167
expected_win_prob   :   0.097
buzz_percent        :  65.808
mean_buzz_position  :   0.417
```

`year` with `category_points` provided the following:

```
accuracy            :  12.256
expected_win_prob   :   0.098
buzz_percent        :  65.897
mean_buzz_position  :   0.420
```

`year` with `subcategory_points` provided the following:

```
accuracy            :  12.256
expected_win_prob   :   0.098
buzz_percent        :  66.075
mean_buzz_position  :   0.419
```

## Best Performance

Turning on Kaitlyn's features & year vector:

```
accuracy            :  14.831
expected_win_prob   :   0.147
buzz_percent        :  40.853
mean_buzz_position  :   1.200
```
