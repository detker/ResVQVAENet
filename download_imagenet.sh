#!/bin/bash

DATA_FOLDER="data"

pip install kaggle unzip

mkdir -p "$DATA_FOLDER"

cat file_list.txt | xargs -n 1 -P 5 -I {} kaggle datasets download -d sautkin/{} -p "$DATA_FOLDER"

for file in $(cat file_list.txt | head -n 4); do
  ZIP_FILE=$(ls "$DATA_FOLDER/$file.zip" | head -n 1)
  if [ -f "$ZIP_FILE" ]; then
      unzip -o "$ZIP_FILE" -d "$DATA_FOLDER/train"
  else
      echo "No zip file found in $DATA_FOLDER."
  fi
done
file=$(cat file_list.txt | tail -n 1)
ZIP_FILE=$(ls "$DATA_FOLDER/$file.zip" | head -n 1)
if [ -f "$ZIP_FILE" ]; then
    unzip -o "$ZIP_FILE" -d "$DATA_FOLDER/test"
else
    echo "No zip file found in $DATA_FOLDER."
fi
