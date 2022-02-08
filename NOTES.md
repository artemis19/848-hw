# Notes

> Group 10, Team SKY | Friday, February 11, 2022
-------------------------------------------------

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

* question_text
* score
* log2(run_length) - smaller number is better!
* category average
* disambiguation in the question
* guess aggregation average

### 