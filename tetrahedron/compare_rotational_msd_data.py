''' 
Quick script to plot rotational msd data from specified pkl files.
usage: 
  python compare_rotational_msd_data.py rot-msd-data-file-1.pkl
    rot-msd-data-file-2.pkl etc.

Currently configured to compare two different MSD series, one tracking
Center of Mass, one tracking a single vertex.
''' 

import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import sys
sys.path.append('..')

from plot_rotational_msd import calculate_zz_and_rot_msd_at_equilibrium
from translational_diffusion_coefficient import calculate_average_mu_parallel_and_perpendicular
import tetrahedron_free as tf
from utils import MSDStatistics
from utils import plot_time_dependent_msd

if __name__ == '__main__':
  # Don't care about paramters here, pass an empty dictionary.
  combined_msd_statistics = MSDStatistics({})
  label_list = ['bug parallel', 'no bug parallel', 'bug perp', 'no bug perp']
#  label_list = [' Parallel MSD Vertex', ' Parallel MSD CoM', ' Perpendicular MSD Vertex', 
#                ' Perpendicular MSD CoM']
  symbol_list = ['o', 'd', 's', '^', '.', '+']
  colors = ['b', 'g', 'r', 'c']
  for k in range(1, len(sys.argv)):
    data_file = sys.argv[k]
    data_name = os.path.join('.', 'data', data_file)
    with open(data_name, 'rb') as f:
      msd_statistics = cPickle.load(f)
      msd_statistics.print_params()
      # HACK, add xx and yy to get translational data
      for scheme in msd_statistics.data:
        for dt in msd_statistics.data[scheme]:
          for j in range(len(msd_statistics.data[scheme][dt][1])):
            msd_statistics.data[scheme][dt][1][j][0][0] = (
              msd_statistics.data[scheme][dt][1][j][0][0] +
              msd_statistics.data[scheme][dt][1][j][1][1])
            msd_statistics.data[scheme][dt][2][j][0][0] = np.sqrt(
              msd_statistics.data[scheme][dt][2][j][0][0]**2 +
              msd_statistics.data[scheme][dt][2][j][1][1]**2)
      figure_indices = [1, 2, 3, 4, 5, 6]
      for l in range(6):
        ind = [l, l]
        if k == 1:
          plot_time_dependent_msd(msd_statistics, ind, figure_indices[l], color=colors[k-1],
                                  label=label_list[(k-1) + min(l, 2)], symbol=symbol_list[l], num_err_bars=200)
        elif k == 2:
          plot_time_dependent_msd(msd_statistics, ind, figure_indices[l], color=colors[k-1],
                                  label=label_list[(k-1) + min(l, 2)], symbol=symbol_list[l],
                                  data_name='COMData-%s-%s.txt' % (l, l),
                                  num_err_bars=200)
 
  average_mob_and_friction = calculate_average_mu_parallel_and_perpendicular(2000)
  [zz_msd, rot_msd] = calculate_zz_and_rot_msd_at_equilibrium(2000)
  translation_end = 360.0
  for l in range(6):
    pyplot.figure(figure_indices[l])
    if l in [0, 1]:
      pyplot.plot([0.0, 150.], 
                  [0.0, 150.*4.*tf.KT*average_mob_and_friction[0]], 
                  'k-', label='Mu Parallel Vertex')

      pyplot.plot([0.0, translation_end], 
                  [0.0, translation_end*2.*tf.KT*0.0711],  #CoM mu.
                  'k--', label='Mu Parallel CoM')
      pyplot.xlim([0.0, translation_end])
      pyplot.ylim([0., translation_end*4.*tf.KT*average_mob_and_friction[0]])
    elif l == 2:
      pyplot.plot([0.0, translation_end], [zz_msd, zz_msd], 'k--', label='Equilibrium Perp MSD')
      pyplot.xlim([0., translation_end])
      pyplot.ylim([0., translation_end*2.*tf.KT*0.0711])
    pyplot.title('MSD(t) for Free Tetrahedron')
    pyplot.legend(loc='best', prop={'size': 10})
    pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s-%s.pdf' % 
                   (l, l))
  
    
    
  
