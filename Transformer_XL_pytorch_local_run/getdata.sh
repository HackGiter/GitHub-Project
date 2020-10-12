echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

echo "- Downloading text8 (Character)"
mkdir -p text8
cd text8
wget --continue http://mattmahoney.net/dc/text8.zip
python ../../prep_text8.py
cd ..

echo "---"
echo "Happy language modeling :)"
