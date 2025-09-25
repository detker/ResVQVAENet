#!/bin/bash

DATA_FOLDER="data"

pip install kaggle unzip

mkdir -p "$DATA_FOLDER"
mkdir -p "$DATA_FOLDER/train"
mkdir -p "$DATA_FOLDER/test"

kaggle datasets download -d ambityga/imagenet100 -p "$DATA_FOLDER"

ZIP_FILE=$(ls "$DATA_FOLDER/imagenet100.zip" | head -n 1)
if [ -f "$ZIP_FILE" ]; then
    unzip -o "$ZIP_FILE" -d "$DATA_FOLDER"
else
    echo "No zip file found in $DATA_FOLDER."
fi

for dir in $(cat file_list1.txt | head -n 4); do
  DIR_LOC="$DATA_FOLDER/$dir"
  if [ -d "$DIR_LOC" ]; then
      mv "$DIR_LOC"/* "$DATA_FOLDER/train"
      rm -rf "$DIR_LOC"
  else
      echo "Error. $DIR_LOC"
  fi
done
dir=$(cat file_list1.txt | tail -n 1)
DIR_LOC="$DATA_FOLDER/$dir"
if [ -d "$DIR_LOC" ]; then
    mv "$DIR_LOC"/* "$DATA_FOLDER/test"
    rm -rf "$DIR_LOC"
else
    echo "Error."
fi
