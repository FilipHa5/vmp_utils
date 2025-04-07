#!/bin/bash

mkdir extracted
mkdir downloads # @TODO: unused dir

while read url
do

    filename_with_ext=${url##*/}
    filename_no_ext=${filename_with_ext%*.tar.xz}

    # Extract vehicle category name (category is the part before digits)
    shopt -s extglob
    category_name=${filename_no_ext%%+([[:digit:]])}

    echo processing "$filename_with_ext"...

    # Download the file, retry if it fails (with a maximum of 3 retries)
    if ! wget --tries=3 --timeout=30 -O "$filename_with_ext" "$url"; then
        echo "Failed to download $filename_with_ext. Skipping..."
        continue
    fi

    # Extract the tar.xz file
    if ! tar -Jxf "$filename_with_ext"; then
        echo "Failed to extract $filename_with_ext. Skipping..."
        continue
    fi

    # Create the category directory inside extracted if it doesn't exist
    if [ ! -d "extracted/${category_name}" ]
    then
        mkdir -p "extracted/${category_name}"
    fi

    # Move the extracted files to the category directory
    if [ -d "$filename_no_ext" ]; then
        mv "$filename_no_ext"/* "extracted/$category_name/"
        # Clean up the extracted folder
        rm -rf "$filename_no_ext"
    else
        echo "No files extracted for $filename_with_ext. Skipping..."
    fi

done < $1 # Input file containing URLs (links.txt)

# Move files from 'tir' to 'truck' and clean up the 'tir' directory
if [ -d "extracted/tir" ]; then
    mv extracted/tir/* extracted/truck/ && rm -rf extracted/tir/
else
    echo "No 'tir' directory found, skipping move to 'truck'."
fi

# Optionally, you could check if there are any remaining issues or directories
echo "Processing complete."
