# check empty entry
grep -l ,, *

# test column 1 is the same
for f in *_LT_* ; do echo $f; diff <(head sorted-ZDJM_3GD_02_201510_LT_BILL.csv | cut -d ',' -f 1) <(head $f | cut -d ',' -f 1); done;
for f in *_AT_* ; do echo $f; diff <(head sorted-ZDJM_3GD_02_201510_AT_BILL.csv | cut -d ',' -f 1) <(head $f | cut -d ',' -f 1); done;

# sort by 1st column
for f in *; do echo $f; sort -t"," -k1n,1 $f > sorted-$f; done;

# check column 1 is the same
for f in *_LT_* ; do echo $f; diff <(cat sorted-ZDJM_3GD_02_201510_LT_BILL.csv | cut -d ',' -f 1) <(cat $f | cut -d ',' -f 1); done;
for f in *_AT_* ; do echo $f; diff <(cat sorted-ZDJM_3GD_02_201510_AT_BILL.csv | cut -d ',' -f 1) <(cat $f | cut -d ',' -f 1); done;

# check number of colum of some line (10)
for f in *; do echo $f; head -n 10 $f | tail -n 1 | tr "," "\n" | wc -l; done;

# check presented status
cat *|cut -d , -f 1 --complement|tr "," "\n"|sort|uniq -c
