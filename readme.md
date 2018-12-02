still under construction...

## environment

python 3.6 / tensorflow 1.4.1 / cuda 8.0

## build extra operators

```
cd operators
chmod +x buildOps.sh
./buildOps.sh
```

## testing

```
wget https://www.dropbox.com/s/nn5lb65cseqdl2w/188_Area_5_office_22.pkl.zip
wget https://www.dropbox.com/s/vr03wnb42ev4ajo/s3dis.zip

mkdir data/s3dis
cd data/s3dis
unzip ../../188_Area_5_office_22.pkl.zip

mkdir model/s3dis
cd data/model/s3dis
unzip ../../s3dis.zip

cd ../.. # move to root dir
python test_model.py
```

The result is written in the "result/*.txt".