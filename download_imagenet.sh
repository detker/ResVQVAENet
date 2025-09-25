#!/bin/bash

DATA_FOLDER="data"

pip install kaggle unzip

mkdir -p "$DATA_FOLDER"

#KAGGLE_DATASET="sautkin/imagenet1k0"
#kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_FOLDER"
#echo 'Part 0 downloaded.'
#
#KAGGLE_DATASET="sautkin/imagenet1k1"
#kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_FOLDER"
#echo 'Part 1 downloaded.'
#
#KAGGLE_DATASET="sautkin/imagenet1k2"
#kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_FOLDER"
#echo 'Part 2 downloaded.'
#
#KAGGLE_DATASET="sautkin/imagenet1k3"
#kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_FOLDER"
#echo 'Part 3 downloaded.'
#
#KAGGLE_DATASET="sautkin/imagenet1kvalid"
#kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_FOLDER"
#echo 'Validation part downloaded. Done.'

cat file_list.txt | xargs -n 1 -P 2 -I {} kaggle datasets download -d sautkin/{} -p "$DATA_FOLDER"

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
#ZIP_FILE=$(ls "$DATA_FOLDER"/imagenet1k0.zip | head -n 1)
#if [ -f "$ZIP_FILE" ]; then
#    unzip -o "$ZIP_FILE" -d "$DATA_FOLDER/train"
#else
#    echo "No zip file found in $DATA_FOLDER."
#fi
#ZIP_FILE=$(ls "$DATA_FOLDER"/imagenet1k1.zip | head -n 1)
#if [ -f "$ZIP_FILE" ]; then
#    unzip -o "$ZIP_FILE" -d "$DATA_FOLDER/train"
#else
#    echo "No zip file found in $DATA_FOLDER."
#fi
#ZIP_FILE=$(ls "$DATA_FOLDER"/imagenet1k2.zip | head -n 1)
#if [ -f "$ZIP_FILE" ]; then
#    unzip -o "$ZIP_FILE" -d "$DATA_FOLDER/train"
#else
#    echo "No zip file found in $DATA_FOLDER."
#fi
#ZIP_FILE=$(ls "$DATA_FOLDER"/imagenet1k3.zip | head -n 1)
#if [ -f "$ZIP_FILE" ]; then
#    unzip -o "$ZIP_FILE" -d "$DATA_FOLDER/train"
#else
#    echo "No zip file found in $DATA_FOLDER."
#fi
#ZIP_FILE=$(ls "$DATA_FOLDER"/imagenet1kvalid.zip | head -n 1)
#if [ -f "$ZIP_FILE" ]; then
#    unzip -o "$ZIP_FILE" -d "$DATA_FOLDER/test"
#else
#    echo "No zip file found in $DATA_FOLDER."
#fi




