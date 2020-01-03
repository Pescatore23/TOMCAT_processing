# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:50:50 2019

@author: firo
"""

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time [s]')
#plt.xlim(1250, 1440)
ax1.set_ylabel('Helmholtz energy [J]', color = color)
ax1.plot(time, F, color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('kinetic energy [J]', color = color)
ax2.plot(time[1:], E_kin[80,:], color = color)
ax2.tick_params(axis = 'y', labelcolor=color)
fig.tight_layout()
plt.title('T3_100_6_label_120')
#plt.show()

fig.savefig(r"H:\03_Besprechungen\Group Meetings\November_2019\_energy_label_120.png", format='png', dpi=600, bbox_inches='tight')