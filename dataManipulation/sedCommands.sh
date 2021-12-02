#!/bin/bash
sed -Ei 's/config\.[a-zA-Z]+/\U&/g' *.py
sed -Ei 's/CONFIG\./\L&/g' *.py

