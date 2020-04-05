from matplotlib import pyplot as plt
import numpy as np
import time

im = plt.imread('/home/chunhung/Downloads/NUH_MB_English_Level_2.jpg')


plt.figure(figsize=(18,18))
plt.ion()

for sim in range(50):
    plt.cla()
    centroid = np.array([[500,120],[520,210],[730,240],[500,400],[480,650],[220,420],[1000,420]])
    all_pts = np.array([[None,None]])
    for c in centroid:
        x_sto = [np.random.normal(50,20) for i in range(20)]
        y_sto = [np.random.normal(50,20) for i in range(20)]    
        choose = np.random.choice(list(range(12,20)))
        x_sto = x_sto[:choose]
        y_sto = y_sto[:choose]    
        all_pts = np.append(all_pts ,np.append(np.expand_dims(np.array(x_sto) + c[0],axis=1),np.expand_dims(np.array(y_sto) + c[1],axis=1),axis=1),axis=0)


    plt.imshow(im,cmap='gray')
    plt.scatter(all_pts[:,0],all_pts[:,1],color='red')
    plt.show()
    plt.pause(0.03)
    if len(str(sim)) == 1:
        print_idx = '00' + str(sim)
    elif len(str(sim)) == 2:
        print_idx = '0' + str(sim)
    else:
        print_idx = str(sim)
    plt.savefig('utils/temp_dir' + '/'+print_idx + '.png')
     

