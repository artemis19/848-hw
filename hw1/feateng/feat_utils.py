from tqdm import tqdm
import json
import math
from collections import OrderedDict
from typing import Iterable, Mapping, Any, List, Tuple
import numpy as np

import qbdata

kSEED = 1701
kBIAS = "BIAS_CONSTANT"


def n_tokens_feature(sent: str):
    return math.log2(len(sent.split()))


def return_train_features(example):
    # This collects features from ALL guesses
    # from ALL the question pool (82395)

    e = example

    disambiguation_points = 0
    if "(" in e["guess"] and ")" in e["guess"]:
        disambiguation = (
            e["guess"].split("(")[1].split(")")[0].replace("_", " ").lower()
        )
        for word in disambiguation.split():
            disambiguation_points += e["question_text"].lower().count(word[:-1])
    if "[" in e["guess"] and "]" in e["guess"]:
        disambiguation = (
            e["guess"].split("(")[1].split(")")[0].replace("_", " ").lower()
        )
        for word in disambiguation.split():
            disambiguation_points += e["question_text"].lower().count(word[:-1])

    category_points = 0
    category_pieces = e["category"].lower()
    for word in category_pieces.split():
        category_points += e["question_text"].lower().count(word[:-1])

    # Widen for root words (chemical/chemistry)
    subcategory_points = 0
    if "subcategory" in e.keys() and isinstance(e["subcategory"], str):
        subcategory_pieces = e["subcategory"].lower()
        for word in subcategory_pieces.split():
            subcategory_points += e["question_text"].lower().count(word[:5])

    return [
        1.0,
        example["score"],
        example["run_length"],
        disambiguation_points,
        category_points,
        subcategory_points,
        # 1 - example["run_length"], # inverse because higher means greater chance
        # math.log2(all_guesses.count(example["guess"])),
    ]


def prepare_train_inputs(
    vocab: List[str], examples: Iterable[Mapping[str, Any]]
) -> Tuple[List[np.ndarray]]:
    """Fill this method to create input features representations and labels for training Logistic Regression based Buzzer.

    :param vocab: List of possible guesses and categories
    :param examples: An iterable of python dicts representing guesses
    across all QANTA example in a dataset. It has the following default schema:
        {
            "id": str,
            "label": str,
            "guess:%s": 1,
            "run_length": float,
            "score": float,
            "category%s": 1,
            "year": int
        }

    You must return the fixed sized numpy.ndarray representing the input features.

    Currently, the function only uses the score a feature along with the bias.
    The logistic regression doesn't implicitly model intercept (or bias term),
    it has to be explicitly provided as one of the input values.
    """

    # all_guesses = [ e["guess"] for e in examples ]

    # inputs = np.array([[1.0, e['score']] for e in examples], dtype=np.float32)
    inputs = np.array([return_train_features(e) for e in examples], dtype=np.float32)

    labels = np.array([e["label"] for e in examples], dtype=int)
    return inputs, labels


def prepare_eval_input(
    vocab: List[str], sub_examples: Iterable[Mapping[str, Any]]
) -> List[np.ndarray]:
    """This function is used during end to end evaluation for computing expected win probability.
    The evaluation is not done just over a logistic regressor, but with the final gold-answer to the question.
    You should assume that the guess with the highest score will be selected as the final prediction,
    but you may use the properties of other guesses to determine the features to the logistic regression model.

    Note: Any label information will explicitly be removed before calling this function.

    :param vocab: List of possible guesses and categories
    :param sub_examples: An iterable of python dicts representing top-k guesses
    of a QANTA example at a particular run length. It has the following default schema:
    {
            "guess:%s": 1,
            "run_length": float,
            "score": float,
            "category%s": 1,
            "year": int
    }
    """

    """
    This is the function where we determine which of these 
    to actually buzz in on. Here we can "compete against our peers"
    """

    all_guesses = [e["guess"] for e in sub_examples]
# print("SUB EXAMPLES")
# # print(len(sub_examples))
# print(all_guesses)
# print("=================================")

    scores = [e["score"] for e in sub_examples]
    score_idx = np.argmax(scores)
    run_length = [e["run_length"] for e in sub_examples]
    rl_idx = np.argmax(run_length)
    tokens_length = [n_tokens_feature(e["question_text"]) for e in sub_examples]
    tokens_idx = np.argmax(tokens_length)

    # disam = []
    # for e in sub_examples:
    #     disambiguation_points = 0
    #     if "(" in e["guess"] and ")" in e["guess"]:
    #         disambiguation = e["guess"].split('(')[1].split(')')[0].replace("_", " ").lower()
    #         for word in disambiguation.split():
    #             disambiguation_points += e["question_text"].lower().count(word[:-1])
    #     disam.append(disambiguation_points)
    # disam_idx = np.argmax(disam)

    # input = np.array([
    #     1.0,
    #     scores[score_idx],
    #     run_length[rl_idx], # highest runtime points
    #     1.0, # disambiguation points constant
    #     1.0, # category points
    #     1.0, # subcategory points
    # ], dtype=np.float32)

    # return input

    e = list(sub_examples)[score_idx]
    return np.array(return_train_features(e), dtype=np.float32)


def make_guess_dicts_from_question(
    question: qbdata.Question,
    runs: List[str],
    runs_guesses: List[List[Tuple[str, float]]],
):
    """Creates an iterable of guess dictionaries from the guesser outputs.
    Feel Free to add more features to the dictionary.
    However, DO NOT add any label specific information as those would be removed explicitly
    and will be considered as breaking the Honor Code.

    :param question: QuizBowl question
    :param runs: a list of question prefixes for the input question
    :param runs_guesses: list of tfidf_guesser outputs for each question_prefix in runs.

    """

    assert len(runs) == len(
        runs_guesses
    ), "'runs_guesses' should have same length as 'runs'"

    for question_prefix, guesses in zip(runs, runs_guesses):
        for raw_guess in guesses:
            page_id, score = raw_guess
            guess = {
                "id": question.qanta_id,
                "guess:%s" % page_id: 1,
                "run_length": len(question_prefix) / 1000,
                "score": score,
                "label": question.page == page_id,
                "category:%s" % question.category: 1,
                "year:%s" % question.year: 1,
                # ============================
                "question_text": question_prefix,
                "guess": page_id,
                "category": question.category,
                "subcategory": question.subcategory,
            }
            yield guess


def write_guess_json(
    guesser: "TfIdfGuesser",
    filename: str,
    questions: Iterable[qbdata.Question],
    run_length: int = 200,
    censor_features=["id", "label"],
    num_guesses: int = 5,
    batch_size=1,
):
    """
    Returns the vocab, which is a list of all features.

    You DON'T NEED TO CHANGE THIS function.

    :param guesser: TfIdfGuesser
    :param filename: path for the output jsonline file
    :param questions: an iterable of Qanta questions
    :param run_length: the difference in characters scanned between consecutive prefixes generated after reading a question.
    :param censor_features: list of features not allowed to use
    :param num_guesses: total number of guesses extracted from the guesser for each question_prefix
    :param batch_size: number of Qanta questions processed at once. Setting this -1 will process all questions at the same time.
    """
    vocab_set = OrderedDict({kBIAS: 1})

    print("Writing guesses to %s" % filename)

    N = len(questions)

    if batch_size == -1:
        batch_size = N  # process everything at once! GO CRAZY! But only do this to iterate over very small set.

    question_batches = [questions[i : i + batch_size] for i in range(0, N, batch_size)]

    with open(filename, "w") as outfile:
        for batch in tqdm(question_batches):

            string_buffer = []

            runs_segments = {}
            all_runs = []
            for ques in batch:
                runs, _ = ques.runs(run_length)
                runs_segments[ques.qanta_id] = len(all_runs), len(runs)
                all_runs.extend(runs)

            batch_runs_guesses = guesser.guess(all_runs, max_n_guesses=num_guesses)

            for ques in batch:
                start_index, guesses_size = runs_segments[ques.qanta_id]
                runs = all_runs[start_index : start_index + guesses_size]
                runs_guesses = batch_runs_guesses[
                    start_index : start_index + guesses_size
                ]
                guesses = make_guess_dicts_from_question(ques, runs, runs_guesses)

                for guess in guesses:
                    for ii in guess:
                        # Don't let it use features that would allow cheating
                        if ii not in censor_features and ii not in vocab_set:
                            vocab_set[ii] = 1
                    string_buffer.append(json.dumps(guess, sort_keys=True))
            outfile.write("\n".join(string_buffer))
            outfile.write("\n")
    print("")
    return [*vocab_set]
