# Geopandas

{% embed url="https://www.kaggle.com/code/keshavramaiah/hotel-recommender/notebook" %}

```python
from sklearn.cluster import KMeans
import numpy as np

wv = metap2v_listing.WPi.weight.detach().cpu().numpy()

# clustering ##################################################
htl_mg_korea = hotel_merge[hotel_merge['CNTRY_CD'] == 'KR']


kr_item_ids = np.asarray(htl_mg_korea['item_idx'])

kritem_2_wvloc = {}
for i, kritemid in enumerate(kr_item_ids):
    kritem_2_wvloc[kritemid] = i
    
kmeans = KMeans(n_clusters=20, random_state=0).fit(wv[kr_item_ids])

htl_mg_korea['cluster'] = htl_mg_korea['item_idx'].map(lambda x: kmeans.labels_[kritem_2_wvloc[x]])
#################################################################

# 
worldmap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
koreamap = worldmap[worldmap['name'] == 'South Korea']
# Creating axes and plotting world map
fig, ax = plt.subplots(figsize=(25, 15))
koreamap.plot(color="lightgrey", ax=ax)

x = htl_mg_korea['LO'].values
y = htl_mg_korea['LA'].values
c = htl_mg_korea['cluster'].values
s = htl_mg_korea['click_count'].values


size_up = 1
scatter = ax.scatter(x, y, c=c, s=s*size_up, alpha=0.6)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="Clusters(random)", fontsize=18, markerscale=3)
plt.setp(legend1.get_title(), fontsize=18)
ax.add_artist(legend1)

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="lower center", title="Clicks", fontsize=12)
plt.setp(legend2.get_title(), fontsize=20)


ax.set_title('Meta-Prod2Vec + booked', fontsize=20)

# ax.set_xlim(96, 127)
# ax.set_ylim(6, 23)

plt.show()

```

<figure><img src="../.gitbook/assets/image (1) (2) (2).png" alt=""><figcaption></figcaption></figure>
