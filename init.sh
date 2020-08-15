pip install torchtext --upgrade

pip install spacy

python -m spacy download fr
python -m spacy download en
python -m spacy download fr_core_news_lg
python -m spacy download en_core_web_lg

mkdir -p ./data
mkdir -p ./checkpoint

wget --no-check-certificate \
    http://www.statmt.org/europarl/v7/fr-en.tgz \
    -O ./data/fr-en.tgz
    
tar -xzvf ./data/fr-en.tgz -C ./data

rm -rf ./data/fr-en.tgz