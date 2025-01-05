rm -rf code.zip
zip -r code.zip mastering_rl/  -x "mastering_rl/__pycache__/*"
cp code.zip _static/
