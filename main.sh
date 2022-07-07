mkdir output
mkdir output/gs
mkdir output/rs

# RS simulator
seq=01
echo $seq
sudo docker run --rm --interactive \
  --user $(id -u):$(id -g)      \
  --volume "$(pwd):/kubric"     \
  kubricdockerhub/kubruntu      \
  /usr/bin/python3 rs/model.py  \
  --use_motion_blur=1           \
  --out_dir=output/rs/seq$seq/    \
  --pose=./rs/data/pose$seq.csv \
  --write=1

# GS simulator

