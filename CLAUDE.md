
Example rsync command for syncing files
rsync -avz -e 'ssh -p 35141 -i ~/.ssh/vastai' \  
  /Users/aszfalt/Projects/research/scene_graph_generation/test_forward_backward.py \
  root@ssh9.vast.ai:~/scene_graph_generation/