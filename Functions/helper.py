import matplotlib.pyplot as plt
import numpy as np

def PCA_order_swap(x):
    #swaps clusters to order from left to right on pca
    #probably should have useed some type of map
    x=int(x)
    y=0
    if x == 5:
         y=0
    elif x == 2:
        y= 1
    elif x== 1:
        y=2
    elif x == 4:
        y=3
    elif x == 0:
        y=4
    else:
        y=5
    return y

def count_clusters(abridge):
    """
    counts how many data points are in each cluster per topology, uses a data frame
    """
    dic={}
    for i in abridge["topology"].unique():
        dic[i]=[]
    for i in range(6):
        Cluster1=abridge[abridge["Cluster"]==i]
        for j in abridge["topology"].unique():
            if sum(Cluster1["topology"]==j) > 0:
                dic[j].append(sum(Cluster1["topology"]==j))
            else:
                dic[j].append(0)
    return dic

def anaylsis(mega,adjust=1,n=1,epochs=500,save=True):
    """_summary_

    Args:
        mega (_type_): _description_
        adjust (int, optional): _description_. Defaults to 1.
        n (int, optional): _description_. Defaults to 1.
        epochs (int, optional): _description_. Defaults to 500.
        save (bool, optional): _description_. Defaults to True.
    """
    for count,i in enumerate(mega):
        hold=[]
        for g in mega[i]:
            scatter_holder=[]
            hold.append(np.array(g))
        fig, axs = plt.subplots(int(len(hold)/2),2, sharex=True,sharey=True)
        rotate=0
        base=0
        m_set=0
        min_set=0
        for z in range(len(hold)):
            scatter_holder.append(hold[z][-1])
            if m_set <= max(-(hold[z]-hold[z][-1])):
                m_set=max(-(hold[z]-hold[z][-1]))
            if min_set <= min(-(hold[z]-hold[z][-1])):
                min_set=min(-(hold[z]-hold[z][-1]))
            axs[base,rotate].fill_between(np.linspace(n,epochs,len(hold[z])),-(hold[z]-hold[z][-1])/adjust, step="post", alpha=0.4,label=f"Base Transfer {z}")
            #axs[base,rotate].scatter(np.linspace(n,epochs,len(hold[z])),-(hold[z]-hold[z][-1]),label=f"Base Transfer {z}")
            fig.suptitle(f"Cluster {count} learning")
            axs[base,rotate].set_title(f"Base Cluster {z}")
            if rotate==1:
                base+=1
                rotate=0
            else:
                rotate+=1
        fig.set_size_inches(18.5, 10.5)
        fig.text(0.04, 0.5, "R^2 deviation from final", va='center', rotation='vertical')
        fig.text(0.5, 0.04, "Epochs", ha='center')
        plt.ylim(min_set/adjust,m_set/adjust)
        if save:
            plt.savefig(f"Learning comp_{i}.png",dpi=400)
        plt.show()
    
def convert_meta(meta):
    """
    converts meta into usable format for analysis
    """
    base={}
    for count,i in enumerate((meta[0])):
        i=meta[0][i]
        base[count]=(i)
    for i in meta[1:]:
        for g in i:
            for count,z in enumerate(i[g]):
                base[g][count]=np.array(base[g][count])+np.array(z)
    return base