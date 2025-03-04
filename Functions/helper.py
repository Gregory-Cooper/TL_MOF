import matplotlib.pyplot as plt
import numpy as np

def PCA_order_swap(x):
    """
    Swaps clusters to order from left to right on PCA.

    Args:
        x (int): The cluster to swap.

    Returns:
        int: The swapped cluster.
    """
    #swaps clusters to order from left to right on pca
    #probably should have used some type of map but had some issue so hard code time
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
    Counts how many data points are in each cluster per topology, using a data frame.

    Args:
        abridge (pd.DataFrame): The data frame containing cluster and topology data.

    Returns:
        dict: A dictionary containing the count of data points in each cluster per topology.
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
    """
    Generates learning curve plots for each cluster.

    Args:
        mega (dict): A dictionary of lists containing cluster data.
        adjust (int, optional): The adjustment factor for the plot. Defaults to 1.
        n (int, optional): The starting epoch for the plot. Defaults to 1.
        epochs (int, optional): The ending epoch for the plot. Defaults to 500.
        save (bool, optional): A flag indicating whether or not to save the plot. Defaults to True.
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
    Converts meta into a usable format for analysis.

    Args:
        meta (list): A list containing the meta data.

    Returns:
        dict: A dictionary containing the converted meta data.
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