To compile the cpp simulation:

```bash
g++ W3Pi_selection.cpp -g $(root-config --glibs --cflags --libs) -o W3Pi_selection.o
end

To run the cpp simulation on one core and dump the result on /data directory:

```bash
taskset -c 0 ./W3Pi_selection.o > ~/pod/thesis/code/scripts-sources/W3Pi-selection-algorithm/data/l1Nano_WTo3Pion_PU200_cppreco.csv
end
