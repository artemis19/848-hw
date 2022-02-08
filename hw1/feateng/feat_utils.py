from tqdm import tqdm
import json
import math
from collections import OrderedDict
from typing import Iterable, Mapping, Any, List, Tuple
import numpy as np

import qbdata
import re

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

categories = ['Religion', 'Mythology', 'Geography', 'Philosophy', 'Science', 'History', 'Literature', 'Fine Arts',
              'Trash', 'Current Events', 'Social Science']
subcategories = ['Mythology Other', 'Mythology Greco-Roman', 'Religion Christianity', 'Geography World',
                 'Science Biology', 'Science Computer Science', 'Fine Arts World', 'Social Science Other',
                 'Social Science American', 'Mythology Indian', 'Geography American', 'History Other',
                 'History American', 'Philosophy European', 'Literature American', 'Science Physics',
                 'Literature Classical', 'Current Events American', 'History British', 'History World',
                 'Literature World', 'Religion Islam', 'Fine Arts Audiovisual', 'History Classical',
                 'Literature European', 'Fine Arts Visual', 'Literature Other', 'Religion East Asian',
                 'Mythology Norse', 'Philosophy East Asian', 'Trash American', 'Religion American', 'Science Other',
                 'Mythology Japanese', 'Religion Other', 'Fine Arts Other', 'Philosophy American', 'Science Math',
                 'Science Chemistry', 'Literature British', 'History European', 'Fine Arts Auditory',
                 'Mythology American']
pages = ['playwright', 'Radiohead_song', 'Mozart', 'play', 'King_of_Thebes', 'Shostakovich', 'poet', 'Mendelssohn',
         'William_Inge_play', '1871', 'Tchaikovsky', 'Chopin_novel', 'musical', 'Mahler', 'New', 'composer',
         'economist', 'dog', 'British_political_party', 'classical_music', 'lawyer', 'thermodynamics', 'element',
         '1775', 'short_story', 'county', 'China', 'actor', 'Animal_Farm', 'abolitionist', 'movie', 'Orff', '1571',
         'music', '1960_film', 'drink', 'tribe', 'essay', 'Beethoven', 'Spinoza', 'Xenophon', 'script', 'chemist',
         'Michelangelo', 'surname', 'UK', 'king', 'god', 'Picasso', 'cosmology', 'Rome_character', 'headgear', 'A',
         'explorer', 'Vermeer', 'howitzer', 'scientist', 'U.S._game_show', 'light', '1748', 'State', 'river',
         '1970_film', 'historian', 'season', '1869', 'lawgiver', 'geology', 'Grieg', 'detective_agency', 'calculus',
         'comics', "Don't_Fear", 'colony', 'John', 'dwarf_planet', 'president', 'filibuster', 'mechanics', 'Prokofiev',
         'politician', 'art', 'writer', 'book', 'Holbein', 'mother_of_Amphion', 'Norris_novel', 'Brahms', '1765',
         'praetor', 'Fall_River,_Massachusetts', 'territorial_entity', 'biology', 'Gogol_short_story', 'deity',
         'author', 'owner', 'Petrarch', 'mythology', 'abstract_data_type', 'Dvořák', 'Renoir', 'TV_series',
         'The_Merchant_of_Venice', 'song', "Don't_Do_It", '1291', 'puppet', 'Debussy', 'programming_language',
         'philosopher', 'ship', 'language', '1814', 'polynomial', 'Bruckner', 'To_Say_Nothing_of_the_Dog',
         '15th_century_BC', 'Haydn', 'periodical', 'ecology', 'emperor', 'novelist', 'U.S._state', 'novel', 'Vivaldi',
         '2003', 'choreographer', 'unit', 'Liszt', 'engineering', 'band', 'physics', 'board_game', 'ancient_kingdom',
         'symbolism', 'Hamlet', '1939', 'Saint-Gaudens', 'Caravaggio', 'ballet', 'United_States',
         'object-oriented_programming', 'U.S._TV_series', 'city', 'Nineteen_Eighty-Four', '1786', 'overture', 'U.S.',
         'Dante', 'computer_science', 'personification', 'number', 'theologian', 'geometry', 'place', 'of_England',
         'The_Tempest', 'mathematics', 'astronomy', 'Paganini', 'BB-39', 'Strauss', 'Shakespeare_poem', 'speech',
         'politics', 'Aristotle', 'Put_a_Ring_On_It', 'location', 'country', 'Scriabin', 'Chopin', 'programming',
         '1099', 'ancestor_of_Noah', 'novella', 'Roman_province', 'painting', 'Great_Expectations', 'region',
         'convention', 'Ireland', 'goddess', 'poetry_collection', 'Ives', 'Sonic_the_Hedgehog', '1992_film', 'planet',
         'film', 'poem', 'psychology', 'opera', 'Byatt_novel', 'data_structure', 'given_name',
         "Alice's_Adventures_in_Wonderland", 'For_Massenet', 'Queen_of_Ethiopia', 'linear_algebra', 'King_Lear',
         'Sibelius', 'constellation', 'Plato', 'psychologist', 'U.S._Constitution', 'anatomy', 'general', 'Harry', 'IV',
         'governor', 'Hobbes_book', 'ACR-1', 'Schubert', 'people', 'scandal', 'statistical_mechanics', 'D', 'botanist',
         'company', 'Adele_song', 'Saint-Saëns', 'kidney', 'economics', 'Latter_Day_Saints', 'philosophy', 'painter',
         'artist', 'Bill', 'mathematics_and_physics', 'video_game', 'With_a_Swing', 'university',
         'biology_and_chemistry', 'separatist_group', 'game', 'consul', 'set', 'James', 'dialogue', 'Genesis',
         'chemistry', 'operating_system', 'series', 'computing', 'moon', 'character', '1997_Austrian_film',
         'exponentiation', 'Serbia', 'state', 'senator', 'Asia', 'quantum_mechanics', 'restaurant', 'waves',
         'metallurgy', 'Augustine', 'being', 'Polish_trade_union', 'tennis', 'biblical_figure', 'phase_transition']
difficulties = ['open', 'regular_college', 'HS', 'regular_high_school', 'MS', 'Open', 'national_high_school',
                'easy_high_school', 'middle_school', 'College', 'hard_college', 'easy_college', 'hard_high_school']
tournaments = ['DEES', 'SCOP MS 6', 'LOGIC', 'Centennial (MD) Housewrite', 'Claude Shannon Memorial Tournament', 'YMIR',
              'Gunpei Yokoi Memorial Open (side event)', 'CMST', 'Chicago Open Visual Arts', 'Illinois Novice',
              'HSAPQ ACF 3', 'Naveed Bork Memorial Tournament', 'St. Anselms and Torrey Pines', 'GRAPHIC',
              'HSAPQ Tournament 9', 'Letras', 'VCU Open (Saturday)', 'Illinois Open', 'MOHIT (Thomas Jefferson)',
              'Science Non-Strosity', 'Missiles of October', 'Harvard Fall Tournament',
              'Weekend of Quizbowl Saturday Event', 'Cardinal Classic XVII', 'BATE', 'Listory', 'OLEFIN', 'Arrabal',
              'U. of Georgia CCC', 'Michigan Fall Tournament', 'Science Monstrosity', 'WUHSAC XI',
              'Illinois Open/(Fall) Terrapin Invitational', 'HSAPQ VHSL Regionals', 'Harvard Fall Tournament VII',
              'WHAQ I', 'We Have Never Been Modern', 'Sivakumar Day Inter-Nationals', 'PADAWAN', 'Historature', 'EMT',
              'Minnesota Open', 'Spring Offensive (history tournament)', 'Fine Arts Common Links', 'Penn Bowl',
              'Maryland Fall', 'MUT', 'QuAC I', 'Cambridge Open', 'Chicago Open Trash', 'HAVOC', 'Scattergories',
              'Jacopo Pontormo (history tournament)', 'Ladue Invitational Sprint Tournament (LIST)', 'MELD', 'GSAC XXV',
              'SCOP MS 5', 'College History Bowl', 'Minnesota Open KLEE Fine Arts', 'PACE NSC', 'WAO II',
              'Berkeley WIT XII', 'Great Lakes Regional Academic Championship (GLRAC)', 'Fall Kickoff Tournament (FKT)',
              'ICCS', 'Maryland Spring Classic', 'Cardinal Classic XVIII', 'WHAQ II', 'Chicago Open Arts',
              'Ladue Invitational Spring Tournament', 'HSAPQ NSC 2', 'Math Monstrosity',
              'Cheyne 1980s American History', 'HSAPQ Tournament 10', 'The Experiment', 'Kentucky Wildcat', 'BHSAT',
              'SHEIKH', 'HSAPQ VHSL Districts', 'SASS', 'Aztlan Cup', 'HSAPQ ACF 2', 'Titanomachy',
              'New Trier Scobol Solo', 'The Experiment II', '3M: Chicago Open History', 'RILKE', 'ACF Novice',
              'MW GSAC XVII', 'Prison Bowl XI', 'HSAPQ VHSL States', 'Geography Monstrosity 2', 'Gaddis I',
              'VCU Open (Sunday)', 'Gorilla Lit', 'Lederberg Memorial Science Tournament 2: Daughter Cell', 'KABO',
              'QUARK', 'HSAPQ Tournament 16', 'Penn-ance', 'LIST (Ladue Invitational Spring Tournament)', 'XENOPHON',
              'Geography Monstrosity', 'St. Louis Open', 'Cane Ridge Revival', 'Aztlan Cup II/Brown UTT/UNC AWET',
              'Oxford Open', 'Chicago Open', 'HFT', 'CALISTO', 'Terrapin', 'Angels in the Architecture', 'VICO',
              'The Bob Loblaw Law Bowl', 'FRENCH', 'Minnesota Undergraduate Tournament (MUT)', 'JAKOB',
              'Collaborative MS Tournament', 'GSAC XXIII', 'HFT X', 'HSAPQ NASAT Tryout Set', 'ILLIAC',
              'Montgomery Blair Academic Tournament (MBAT)', 'Tyrone Slothrop Literature Singles', 'MYSTERIUM',
              'Harvard International', 'MLK', 'Chipola Lit + Fine Arts', 'VCU Open', 'Virginia Wahoo War', 'IMSAnity 5',
              'Maggie Walker GSAC XIX', 'THUNDER', 'DRAGOON', 'FKT', 'HAVOC II', 'NASAT', 'GDS Ben Cooper Memorial',
              'Prison Bowl', 'UIUC High School Solo', 'VETO', 'Gaddis II', 'Toby Keith Hybrid',
              'Chicago Open Literature', 'MSU/UD Housewrite', 'FIST', 'Maggie Walker GSAC XV', 'BDAT I', 'Law Bowl',
              'STIMPY', "BISB (Brookwood Invitational Scholars' Bowl)", 'RAVE', 'HFT XIV', 'GRAB BAG', 'Delta Burke',
              'Collegiate Novice', 'ACF Winter', 'The New Tournament at Cornell', 'Scattergories 2', 'PIANO',
              'Chicago Open History Doubles', 'A Bit of Lit', "Virginia J'ACCUSE!", 'Cheyne American History People',
              'Vanderbilt ABC/2011 VCU Winter', 'Oxford Online', 'Matt Cvijanovich Memorial Novice Tournament',
              'The Unanswered Question', 'HFT XI', 'Words and Objects', 'NHBB College Nationals',
              'History Doubles at Chicago Open', '(This) Tournament is a Crime', 'Maggie Walker GSAC', 'FICHTE', 'NTV',
              'Minnesota Open Lit', 'Sun n Fun', 'WAO', 'Scobol Solo', 'HSAPQ VHSL Regular Season', 'HSAPQ NSC 1',
              'WELD', 'Fernando Arrabal Tournament of the Absurd', 'Christmas Present', 'FACTS',
              'Illinois Open Literature Tournament', 'Virginia Open', 'Maryland Spring', 'Fall Kickoff Tournament',
              'Minnesota Open Lederberg Memorial Science Tournament', 'WUHSAC VIII',
              'Richard Montgomery Blair Academic Tournament', 'RMBCT', "It's Lit", 'Jordaens Visual Arts',
              'Fall Novice', 'DAFT', 'Illinois Earlybird', 'ANFORTAS', 'Tree of Clues', 'BELLOCO', 'HORROR 1',
              'Terrapin Invitational', 'RAPTURE', 'RMP Fest', 'BELFAST Arts', 'FEUERBACH', 'Tyrone Slothrop Lit',
              'Chicago Open Lit', 'Geography Monstrosity 4', 'A Culture of Improvement', 'SCOP Novice', 'Missouri Open',
              'EFT', 'Chitin', 'SCOP 3', 'George Oppen', 'MASSOLIT', 'WUHSAC IX', 'ACF Nationals', 'HSAPQ 4Q2',
              "Sun 'n' Fun", 'BASK', 'WIT', 'HFT XII', 'CLEAR II', 'TJ NAREN', 'THUNDER II', 'ANGST',
              'Delta Burke 2013', 'HSAPQ Tournament 11', 'SMT', 'HSAPQ National History Bowl',
              'Terrapin Invitational Tournament', 'Bulldog High School Academic Tournament (BHSAT)', "Schindler's Lit",
              'HSAPQ Colonia 2', 'Brookwood Invitational Scholars Bowl', 'NTSS', 'Julius Civilis Classics Tournament',
              'NNT', 'HFT XIII', 'Teitler Myth Singles', 'ACF Regionals', 'Illinois Fall Tournament', 'LIST', 'T-Party',
              'Peaceful Resolution', 'Mahfouz Memorial Lit', 'Zot Bowl', 'MAGNI', 'HSAPQ Tournament 15', 'HSAPQ 4Q1',
              'Philly Cheesteak', 'Maggie Walker GSAC XVIII', 'Minnesota Novice Set', 'Maggie Walker GSAC XVI',
              'Michigan Artaud', '"stanford housewrite"', 'HSAPQ ACF 1', 'BISB', 'The Questions Concerning Technology',
              'Wild Kingdom', 'FILM', 'SACK', 'Michigan Manu Ginobili Open',
              'Mavis Gallant Memorial Tournament (Literature)', 'Crusader Cup', 'Princeton Buzzerfest',
              'Michigan Auspicious Incident', 'Cheyne American History', 'Early Fall Tournament (EFT)',
              'Sun God Invitational', 'SUBMIT', 'Ohio State/VCU housewrite', 'JAMES', 'Western Invitational Tournament',
              'BARGE', 'Chicago John Stuart Mill', 'Masonic', 'Cheyne American Thought', 'Guerrilla at ICT',
              'WORLDSTAR', 'Chicago Open History', 'From Here To Eternity', 'History Bee Nationals', 'Illinois Fall',
              'HSAPQ 4Q 1', 'HSAPQ Tournament 17', 'VCU Closed', 'HSAPQ Tournament 8', 'Illinois Wissenschaftslehre',
              'ACF Fall', 'Spartan Housewrite']
years = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
         2015, 2016, 2017, 2018, 2019, 2020]

def n_tokens_feature(sent: str):
    return math.log2(len(sent.split()))

def return_0_1_features(feat_name, feat_list):
    feat_vec = [0] * (len(feat_list) + 1)
    if feat_name in feat_list:
        feat_vec[feat_list.index(feat_name)] = 1
    else:
        feat_vec[-1] = 1
    return feat_vec

def return_train_features(example):
    """Created this to add code or manipulations to features before passing them into the inputs list."""
    e = example

    disambiguation_points = 0
    if "(" in e["guess"] and ")" in e["guess"]:
        disambiguation = (
            e["guess"].split("(")[1].split("(")[0].replace("_", " ").lower()
        )
        for word in disambiguation.split():
            disambiguation_points += e["question_text"].lower().count(word[:-1])

    category_points = 0
    category_pieces = e["category"].lower()
    for word in category_pieces.split():
        category_points += e["question_text"].lower().count(word[:-1])

    subcategory_points = 0
    if "subcategory" in e.keys() and isinstance(e["subcategory"], str):
        subcategory_pieces = e["subcategory"].lower()
        for word in subcategory_pieces.split():
            subcategory_points += e["question_text"].lower().count(word[:5])

    category_vec = return_0_1_features(e["category"], categories)
    subcategory_vec = return_0_1_features(e["subcategory"], subcategories)
    difficulty_vec = return_0_1_features(e["difficulty"], difficulties)
    tournamemt_vec = return_0_1_features(e["tournament"], tournaments)
    year_vec = return_0_1_features(e["year"], years)


    page_vec = [0] * (1 + len(pages) + 1)
    if e["page"].find('(') != -1 and e["page"].find(')') != -1:
        page_name = re.search(r'\((.*?)\)',e["page"]).group(1)
        if page_name in pages:
            page_vec[1+ pages.index(page_name)] = 1
        else:
            page_vec[-1] = 1
    else:
        page_vec[0] = 1


    return [
        1.0,
        e["score"],
        e["run_length"],
        n_tokens_feature(e["question_text"]),
        *category_vec,
        *subcategory_vec,
        *difficulty_vec,
        *tournamemt_vec,
        *page_vec,
        *year_vec,
        disambiguation_points,
        category_points,
        subcategory_points,
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
    scores = [e["score"] for e in sub_examples]
    score_idx = np.argmax(scores)
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
                # --------------------------------
                "question_text": question_prefix,
                "guess": page_id,
                "page": question.page,
                "difficulty": question.difficulty,
                "category": question.category,
                "subcategory": question.subcategory,
                "tournament": question.tournament,
                "year": question.year,
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
