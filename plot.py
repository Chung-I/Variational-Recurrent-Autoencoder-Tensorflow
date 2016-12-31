import matplotlib.pyplot as plt
import json
import sys
import numpy as np
step_size = 200


def loss_dict_to_arr(stat_dict, step_size):
  curr_step = step_size
  stat_arr = []
  none_combo = 0
  while none_combo < 10:
    loss_val = stat_dict.get(str(curr_step))
    stat_arr.append(loss_val)
    if loss_val:
      none_combo = 0
    else:
      none_combo += 1
    curr_step += step_size
  return stat_arr

def mask_and_plot(np_arrs, labels, value_text, title_name, fig_name):
  #max_step = max([np_arr.size for np_arr in np_arrs])
  plots = []
  for idx, np_arr in enumerate(np_arrs):
    x = np.arange(step_size, step_size * np_arr.size + 1, step_size)
    mask = np.isfinite(np_arr)
    single_plot, = plt.plot(x[mask],np_arr[mask],label=labels[idx])
    plots.append(single_plot)
  plt.legend(loc='upper left', handles=plots)
  plt.ylabel(value_text)
  plt.xlabel('step')
  plt.title(title_name)
  plt.savefig(fig_name)

def jsons2arrs(file_names, value_name):
  arrs = []
  for stat_file_name in file_names:
    with open(stat_file_name, "r") as stat_file:
      stats = json.load(stat_file)
      value_dict = stats[value_name]
      value_arr = loss_dict_to_arr(value_dict, step_size)
      value_arr = np.array(value_arr).astype(np.float32)
      arrs.append(value_arr)
  return arrs


value_arrs = jsons2arrs(sys.argv[3:], sys.argv[1])
labels = [json_name.split(".")[:-1] for json_name in sys.argv[3:]]
mask_and_plot(value_arrs, labels, sys.argv[1], sys.argv[1], sys.argv[2])

