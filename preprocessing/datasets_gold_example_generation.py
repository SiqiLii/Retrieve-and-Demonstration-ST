#Construct datasets that prepend gold example to rare-word-tst split, rare-word-dev split, tst-COMMON split respectively, to form tst_ex split, dev_ex split, tst-COMMON_ex
#python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT}

#Construct train_ex split that each sentence in the reduced-train split is prepended sentences that contains the same sentence-level rare word from the reduced-train split
#python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT} --train True