#!/bin/bash

CHECKPOINT="model_best.pth.tar"


while true; do

    # find any interrupted commands
    cmd="$(grep -P -m 1 '^[^#].*(#\ *started)' commands.txt)"
    if [[ -z "$cmd" ]]; then # cmd is empty
        # find new command to run
        cmd="$(grep -P -m 1 '^[^#].*(\n$)' commands.txt)"
        if [[ -z "$cmd" ]]; then # still no commands
            break # no more commands to run
        fi
        # set started
        python sparse_to_dense/main.py $cmd 2>>tracebacks.txt
    else
        cd sparse_to_dense
        output_folder="$(python -c 'import utils; print(utils.get_output_directory(utils.parse_command()));' $cmd)"
        cd ..

        # check if there is a checkpoint available
        checkpoint_path="$output_folder/$CHECKPOINT"
        if [[ -f "$checkpoint_path" ]]; then
            python sparse_to_dense/main.py --resume $checkpoint_path 2>>tracebacks.txt
        else
            python sparse_to_dense/main.py $cmd 2>>traceback.txt
        fi
    fi

    retval=$?
    if [ $retval -ne 0 ]; then
        # set failed
    else
        # set finished
    fi
done