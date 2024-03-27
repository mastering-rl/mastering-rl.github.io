rm -rf code.zip
zip -r code.zip python_code/  -x "python_code/__pycache__/*"
cp code.zip _static/
