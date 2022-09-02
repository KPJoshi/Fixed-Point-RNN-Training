mkdir usps10
cd usps10
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
bunzip2 usps.bz2
bunzip2 usps.t.bz2
mv usps train.txt
mv usps.t test.txt
cd ..
python3 process_usps.py
rm usps10/*.txt

