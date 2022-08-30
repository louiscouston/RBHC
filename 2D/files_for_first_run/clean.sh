# Use for postprocessing of hc.his initiated via genhpts.f90
# Read number of probes
np=`awk 'NR==1{print $1; exit}' hc.his`
echo $np probes in total
# Remove first line
awk NR\>1 hc.his > tmp.dat
# Write list of probe positions in probes.dat (nx+nz)*2*3 (3 probes/boundary point)
nps=792
echo $nps boundary probes
npb=$((np-nps))
npb1=$((npb+1))
echo $npb bulk probes
awk "NR >= 1 && NR <= $npb" tmp.dat > probes_bulk.dat
awk "NR > $npb && NR <= $np" tmp.dat > probes_side.dat
# Remove probe positions
awk NR\>$np tmp.dat > tmp2.dat
# Write data in data.dat 
awk -vn=$npb -vm=$np 'NR<=i{next}; (NR-i)%m==1{c=1}; c++<=n' tmp2.dat > data_bulk.dat
awk -vn=$nps -vm=$np -vi=$npb 'NR<=i{next}; (NR-i)%m==1{c=1}; c++<=n' tmp2.dat > data_side.dat
# Split velocity and temperature data if necessary
awk '{ print $4 }' data_bulk.dat > t_bulk.dat
awk '{ print $4 }' data_side.dat > t_side.dat
awk '{ print $2 }' data_bulk.dat > u_bulk.dat
awk '{ print $2 }' data_side.dat > u_side.dat
awk '{ print $3 }' data_bulk.dat > v_bulk.dat
awk '{ print $3 }' data_side.dat > v_side.dat
# Write list of times for reference
awk '{ print $1 }' data_side.dat > tmp.dat
awk "NR%$nps==1" tmp.dat > time.dat
echo 'Done'
rm tmp.dat
rm tmp2.dat
