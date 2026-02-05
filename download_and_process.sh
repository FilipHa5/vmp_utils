#!/bin/bash

# make directories
if [ ! -d "extracted" ]; then
    mkdir -p "extracted"
fi
if [ ! -d "downloads" ]; then
    mkdir -p "downloads"
fi

while read url
do

    filename_with_ext=${url##*/}
    target_download_path=downloads/${url##*/}
    filename_no_ext=${filename_with_ext%*.tar.xz}

    # Extract vehicle category name (category is the part before digits)
    shopt -s extglob
    category_name=${filename_no_ext%%+([[:digit:]])}

    # If file does not exist in download perform full procedure
    if [ ! -f $target_download_path ]; then
        # Download the file, retry if it fails (with a maximum of 3 retries)
        if ! wget --tries=3 --timeout=30 -O "$target_download_path" "$url"; then
            echo "Failed to download $filename_with_ext. Skipping..."
            continue
        fi
        
        echo "Extracring $target_download_path file..."
        if ! tar -Jxf "$target_download_path"; then
            echo "Failed to extract $target_download_path. Skipping..."
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
    else
        echo "File $target_download_path exists, skipping..."
    fi


done < $1 # Input file containing URLs (links.txt)

# Move files from 'tir' to 'truck' and clean up the 'tir' directory
if [ -d "extracted/tir" ]; then
    mv extracted/tir/* extracted/truck/ && rm -rf extracted/tir/
else
    echo "No 'tir' directory found, skipping move to 'truck'."
fi

# temporary solution - move incorrectly labeled data to correct dir. TODO.
(cd extracted && source ../tmp_tidy.sh)

echo "Processing complete."
