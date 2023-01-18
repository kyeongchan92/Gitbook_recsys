# stacked bar chart

&#x20;

<figure><img src="../.gitbook/assets/image (24).png" alt=""><figcaption></figcaption></figure>

{% code overflow="wrap" lineNumbers="true" %}
````python
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))
ax = plt.subplot2grid((1,1), (0,0))

x = list(range(20))

width=0.7

## bottom
bottom_y = np.random.randint(10, 100, len(x))
ax.bar(x, bottom_y, width, color='g', lw = 0.5, linestyle='-', edgecolor ='k', label='bottom')
## up
up_y = np.random.randint(10, 100, len(x))
ax.bar(x, up_y, width, bottom=bottom_y, color='r', lw = 0.5, linestyle='-', edgecolor ='k', label='up')
# # text
# for rec_i in range(len(count_df)):
#     reached_rec = list(ax.patches)[:len(count_df)][rec_i]
#     not_reached_rec = list(ax.patches)[len(count_df):][rec_i]
#     ax.text(x = reached_rec.get_x() + reached_rec.get_width()/2,
#                 y = reached_rec.get_height() + not_reached_rec.get_height() + 3000,
#                 s =  f"{reached_rec.get_height() + not_reached_rec.get_height():,}",
#                ha = 'center')


ax2 = ax.twinx()
# member percentage plot
bottom_ratio = bottom_y / (bottom_y + up_y)
line = ax2.plot(x, bottom_ratio, marker='o', color='g')
#text
# for x_pos, y_pos in line[0].get_xydata():
#     plt.text(x_pos+0.1, y_pos-0.008, f'{y_pos:.2f}')

    
ax2.set_ylim((0,1))
ax.set_xticks(x, fontsize=18)
# ax.set_xticklabels(x, fontsize=10)
ax.set_title(f"Your title" ,fontsize=18)
ax.set_xlabel("Your x label",fontsize=18)
ax.set_ylabel('Your y label',fontsize=18)

ax.legend(fontsize=13, loc='upper right')

plt.show()
```
````
{% endcode %}



<figure><img src="../.gitbook/assets/image (12) (2).png" alt=""><figcaption></figcaption></figure>
