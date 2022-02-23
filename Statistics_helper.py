# Draft of statistical comparisons
def compare_layer(l1,l2):
    w1=l1.weight
    w2=l2.weight
    MAE=sum(abs(sum(w1-w2)))
    MSE= sum((w1-w2)**2)/len(w1)
    return MAE,MSE
