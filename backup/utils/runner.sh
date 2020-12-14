#!/bin/bash

# for (( i=0 ; i<80 ; i++ )); do
#     python3 preprocessing.py USE-Large mutated 300
# done

echo "EXP 1"
date

fake=neg
for embedding in glove USE USE-Large; do
    for arch in FC CNN LSTM; do
        cmd="python3 main.py $fake $embedding $arch"
        echo $cmd
        $cmd
    done
done

echo "EXP 2"
date

fake=mutate
for embedding in glove USE USE-Large; do
    for arch in FC CNN LSTM; do
        for extra in add delete replace; do
            cmd="python3 main.py $fake $embedding $arch --extra=$extra"
            echo $cmd
            $cmd
        done
    done
done

echo "EXP 3"
date
fake=neg
for arch in FC CNN LSTM; do
    embedding=InferSent
    cmd="python3 main.py $fake $embedding $arch"
    echo $cmd
    $cmd
done

echo "EXP 4"
date
fake=mutate
for arch in FC CNN LSTM; do
    for extra in add delete replace; do
        embedding=InferSent
        cmd="python3 main.py $fake $embedding $arch --extra=$extra"
        echo $cmd
        $cmd
    done
done

date

echo "DONE!!!"
    # python3 main.py mutate USE CNN --extra=replace
    # python3 main.py mutate USE CNN --extra=replace

# for (( i=0 ; i<40000 ; i++ )); do
#     python3 preprocessing.py USE mutated 2000
# done

# skip=0
# for (( i=0 ; i<40000 ; i++ )); do
#     python3 preprocessing.py InferSent story 300 --skip=$skip
#     if [[ $? != 0 ]]; then
#         skip=$((skip+300))
#     fi
# done

# rm `ls -Sh | tail`


# num=25
# for (( i=0 ; i<40000 ; i++ )); do
#     echo $num
#     python3 preprocessing.py InferSent story $num
#     if [[ $? != 0 ]]; then
#         num=$((num/5))
#     else
#         num=$((num*5))
#     fi
#     # if (( $num > 64 )); then
#     #     num=64
#     # fi
#     if (( $num < 1 )); then
#         num=1
#     fi
# done

# for (( i=0 ; i<40 ; i++ )); do
#     # python3 preprocessing.py InferSent story 1
#     python3 preprocessing.py InferSent story 5
#     python3 preprocessing.py InferSent story 10
#     python3 preprocessing.py InferSent story 20
#     python3 preprocessing.py InferSent story 30
#     python3 preprocessing.py InferSent story 50
#     # python3 preprocessing.py InferSent mutated 30
# done

