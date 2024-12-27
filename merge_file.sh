export nshard=1
export split=valid
export lab_dir=data_mau/public_data/km_label
export n_clusters=100

for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km

for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done > $lab_dir/dict.km.txt