#!/bin/bash

if [ -d "./hdf5" ] 
then
    echo "hdf5 directory already exists"
else
    echo "Creating hdf5 directory"
    mkdir "./hdf5"
fi

if [ -d "./hdf5/hcd" ] 
then
    echo "hdf5/hcd directory already exists"
else
    echo "Creating hdf5/hcd directory"
    mkdir "./hdf5/hcd"
fi

if [ -f "./hdf5/hcd/prediction_hcd_train.hdf5" ] 
then
    echo "hdf5/hcd/prediction_hcd_train.hdf5 file already exists"
else
    curl --header "Host: s3-eu-west-1.amazonaws.com" --header "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36" --header "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header "Accept-Language: en-US,en;q=0.9,sv;q=0.8" --header "Referer: https://figshare.com/" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/24635459/prediction_hcd_train.hdf5" -L -o "./hdf5/hcd/prediction_hcd_train.hdf5"
fi

if [ -f "./hdf5/hcd/prediction_hcd_test.hdf5" ] 
then
    echo "hdf5/hcd/prediction_hcd_test.hdf5 file already exists"
else
    curl --header "Host: s3-eu-west-1.amazonaws.com" --header "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36" --header "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header "Accept-Language: en-US,en;q=0.9,sv;q=0.8" --header "Referer: https://figshare.com/" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/24635438/prediction_hcd_ho.hdf5" -L -o "./hdf5/hcd/prediction_hcd_test.hdf5"
fi

if [ -f "./hdf5/hcd/prediction_hcd_valid.hdf5" ] 
then
    echo "hdf5/hcd/prediction_hcd_valid.hdf5 file already exists"
else
    curl --header "Host: s3-eu-west-1.amazonaws.com" --header "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36" --header "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header "Accept-Language: en-US,en;q=0.9,sv;q=0.8" --header "Referer: https://figshare.com/" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/24635442/prediction_hcd_val.hdf5" -L -o "./hdf5/hcd/prediction_hcd_valid.hdf5"
fi
