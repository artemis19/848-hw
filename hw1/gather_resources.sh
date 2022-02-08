#!/bin/bash

wget "https://848hw-hw1.s3.amazonaws.com/tfidf.pickle"
mv tfidf.pickle models/

wget "https://848hw-hw1.s3.amazonaws.com/qanta.train.json"
mv qanta.train.json custom_data/
