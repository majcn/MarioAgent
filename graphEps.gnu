set datafile separator ","
set terminal postscript enhanced eps solid colour "Helvetica" 22
set output 'plot.eps'

set multiplot
set xlabel "Epizoda"
set ylabel "Vrednost"
set key out right top

plot 'output.out' u 1 title 'NAGRADA', \
     'output.out' u 2 title 'POT'
