#!/bin/bash

cd /root/autodl-tmp/AdaptCLIP


echo "================================"
echo "Start M -> V experiment"
echo "================================"


bash /root/autodl-tmp/AdaptCLIP/scripts/test_mvtec_visa_fewshot.sh

if [ $? -eq 0 ]; then

    echo "================================"
    echo "M -> V finished successfully"
    echo "Start V -> M experiment"
    echo "================================"

    bash /root/autodl-tmp/AdaptCLIP/scripts/test_visa_mvtec_fewshot.sh

else

    echo "M -> V failed, stop."

    exit 1

fi


echo "================================"
echo "All cross-domain experiments finished"
echo "================================"