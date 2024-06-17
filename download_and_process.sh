#!/bin/bash

mkdir extracted
mkdir downloads

while read url
do

    filename_with_ext=${url##*/}
    filename_no_ext=${filename_with_ext%*.tar.xz}
    shopt -s extglob
    category_name=${filename_no_ext%%+([[:digit:]])}

    echo processing "$filename_with_ext"
    wget -O "$filename_with_ext" "$url" &&
    tar -Jxf "$filename_with_ext"

    if [ ! -d "extracted/${category_name}" ]
    then
        mkdir -p "extracted/${category_name}"
    fi

    mv "${filename_no_ext}"/* "extracted/$category_name" &&
    rm -rf "$filename_no_ext" "$filename_with_ext"

done < $1