path1 = "C:\\Users\\janek\\Development\\Git\\Prag\\deep-learning-lecture\\05_cnns_2\\logs-segmentation\\cags_segmentation.py-2022-03-19_174704-bn=True,bs=50,d=None,e=50,ft=True,l=0.0,lr=0.001,lrf=0.0001,ll=warning,s=42,t=1\\cags_segmentation.txt"
path2 = "C:\\Users\\janek\\Development\\Git\\Prag\\deep-learning-lecture\\05_cnns_2\\cags_segmentation.txt"
file1 = open(path1, 'r')
file2 = open(path2, 'r')
for i, (l1, l2) in enumerate(zip(file1.readlines(), file2.readlines())):
    if l1 != l2:
        print(f"{i + 1}. line not equal")
        print("line 1:", l1)
        print("line 2:", l2)