#!/bin/bash
#sed -Ei 's/config\.[a-zA-Z]+/\U&/g' *.py
#sed -Ei 's/CONFIG\./\L&/g' *.py
touch imports.txt
cd dataManipulation
find -name '*.py' -exec sed -n '/import /p' >../imports.txt {} + 
cd ../configurations
find -name '*.py' -exec sed -n '/import/p' >>../imports.txt {} + 
cd ../main
find -name '*.py' -exec sed -n '/import/p' >>../imports.txt {} + 
cd ..
cat imports.txt | awk '!a[$0]++' > uniqueimports.txt
sed '/^#/d' uniqueimports.txt > imports.py
rm imports.txt uniqueimports.txt 
