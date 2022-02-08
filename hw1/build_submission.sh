#!/bin/bash

# Put all the files in a temporary directory
directory=$(mktemp -d)
cp -R feateng/ ${directory}

# Make the directory structure in the temp dir
mkdir ${directory}/models
mkdir ${directory}/custom_data
cp models/guess.vocab ${directory}/models/guess.vocab
cp models/lr_buzzer.pickle ${directory}/models/lr_buzzer.pickle
cp requirements.txt ${directory}

# Build script to pull down model files
cat << EOF > ${directory}/gather_resources.sh
#!/bin/bash

wget "https://848hw-hw1.s3.amazonaws.com/tfidf.pickle"
mv tfidf.pickle models/

wget "https://848hw-hw1.s3.amazonaws.com/qanta.train.json"
mv qanta.train.json custom_data/
EOF

# Zip up the files and put it here
pushd ${directory}
zip submission.zip -r *
popd
mv ${directory}/submission.zip .
