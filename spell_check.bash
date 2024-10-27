for f in intro/*.md; do aspell -c -l en-uk -t $f; done
for f in single-agent/*.md; do aspell -c -l en-uk -t $f; done
for f in multi-agent/*.md; do aspell -c -l en-uk -t $f; done
for f in appendix/*.md; do aspell -c -l en-uk -t $f; done
